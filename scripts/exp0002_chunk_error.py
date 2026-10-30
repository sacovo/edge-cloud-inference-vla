"""EXP-0002 post-hoc (not gating): how fast does a teacher's plan go stale? L1 per chunk step j on held-out demos.

Chunk step j at time t predicts the demo action at t + j, so j is the plan's age when the student uses it (at update
period k with delay k the student uses j = k .. 2k - 1). Teacher-forced: the teacher sees the demo's own proprio and
torque, so this is a lower bound on closed-loop staleness. Same held-out split (last 2 kept demos per task), action
units and per-dimension mean as train.py's val L1 (j = 0 reproduces it). Reference: persistence, i.e. predicting
a_{t+j} = a_t (how much the demo action itself changes over j steps).

Runs as a job (needs torch + the ecvla env):  python scripts/exp0002_chunk_error.py --out <json>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ecvla.policy import ACT_DIM, FastPolicy, PolicyConfig
from ecvla.train import Batcher, EpisodeData

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
DATA = S / "a1/prepared_v2"
SUITES = ("libero_spatial", "libero_object")
SEEDS = (0, 1, 2)
TEACHERS = {8: lambda su, sd: S / f"a1/runs/grid/{su}/none_task-id_10M_gru-l1_s{sd}",
            40: lambda su, sd: S / f"exp0002/runs/teachers/{su}/task-id_10M_h40_s{sd}"}
GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}


@torch.no_grad()
def chunk_error(run_dir: Path, suite: str) -> dict:
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = FastPolicy(PolicyConfig(**ckpt["info"]["policy"])).eval()
    model.load_state_dict(ckpt["state_dict"])
    H = model.cfg.chunk
    data = EpisodeData(DATA, [suite], val_per_task=2)
    assert data.tasks == ckpt["tasks"], "task order differs from the teacher's training data"
    batcher = Batcher(data, ckpt["stats"], "task-id", torch.device("cpu"), H)
    std = torch.tensor(ckpt["stats"]["actions"][1])
    val = data.split(True)
    err, pers, n = torch.zeros(H, ACT_DIM), torch.zeros(H, ACT_DIM), torch.zeros(H)
    for i in range(0, len(val), 8):
        b, target, mask = batcher(val[i: i + 8], np.random.default_rng(0))
        f, _ = model.features(b)
        pred = model.act(f)  # (B, T, H, 7), normalized
        m = mask[..., None].float()  # (B, T, H, 1)
        err += ((pred - target).abs() * std * m).sum((0, 1))
        pers += ((target[:, :, :1] - target).abs() * std * m).sum((0, 1))
        n += mask.float().sum((0, 1))
    err, pers = err / n[:, None], pers / n[:, None]
    out = {"run": str(run_dir), "chunk": H, "val_episodes": len(val), "n_per_j": n.int().tolist(),
           "l1": err.mean(1).tolist(), "persistence_l1": pers.mean(1).tolist()}
    for g, sl in GROUPS.items():
        out[f"l1_{g}"] = err[:, sl].mean(1).tolist()
        out[f"persistence_l1_{g}"] = pers[:, sl].mean(1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rows = []
    for H, path in TEACHERS.items():
        for su in SUITES:
            for sd in SEEDS:
                r = {"horizon": H, "suite": su, "seed": sd, **chunk_error(path(su, sd), su)}
                rows.append(r)
                print(H, su, sd, " ".join(f"{x:.3f}" for x in r["l1"][::4]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
