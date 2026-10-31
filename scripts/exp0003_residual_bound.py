"""EXP-0003 §6 (not gating): how much of the residual bound do the `track` students use?

For each student, on held-out demos (last 2 kept demos per task, as train.py's val split) and per update period k with
the evaluation delay (exactly k), the executed first action's residual r_b = bound * tanh(r / bound) after the first
plan has arrived. Reported per arm group (pos, rot): fraction of (step, dim) at >= 95% and >= 50% of the bound, and
mean / p95 |r_b| in raw action units (bound = 0.5 raw); gripper (unbounded): mean |r| raw. Also the first-action L1
(raw units) of the student and of the plan alone (plan[age], what `replay` executes). Teacher-forced: the student sees
the demo's proprio and torque, so closed-loop residuals can differ.

Runs as a job (needs torch + the ecvla env):  python scripts/exp0003_residual_bound.py --out <json>
"""

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

import ecvla.train as tr
from ecvla.policy import ACT_DIM, FastPolicy, PolicyConfig
from ecvla.train import Batcher, EpisodeData

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
DATA = S / "a1/prepared_v2"
RUNS = S / "exp0003/runs/students"
SUITES = ("libero_spatial", "libero_object")
SIZES = ("10k", "100k")
SEEDS = (0, 1, 2)
KS = (1, 4, 10, 20)
GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6)}


@torch.no_grad()
def residuals(run_dir: Path, suite: str) -> dict:
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ckpt["info"]["policy"])
    assert cfg.track
    model = FastPolicy(cfg).eval()
    model.load_state_dict(ckpt["state_dict"])
    data = EpisodeData(DATA, [suite], val_per_task=2, latent_dir=ckpt["info"]["args"]["latent_dir"])
    assert data.tasks == ckpt["tasks"], "task order differs from the student's training data"
    batcher = Batcher(data, ckpt["stats"], "z-teacher", torch.device("cpu"), cfg.chunk, track=True)
    std = torch.tensor(ckpt["stats"]["actions"][1])
    bound = torch.tensor(cfg.track_bound)
    val = data.split(True)
    out = {"run": str(run_dir), "d": cfg.d, "head_params": ckpt["info"]["head_params"], "val_episodes": len(val)}
    orig = tr.state_source
    try:
        for k in KS:
            tr.state_source = functools.partial(orig, k=k, train=False)
            rb_all, rg_all, l1_s, l1_p = [], [], [], []
            for i in range(0, len(val), 8):  # the eval delay (exactly k) has no jitter: one pass per k
                b, target, mask = batcher(val[i: i + 8], np.random.default_rng(0))
                f, _ = model.features(b)
                r = model.head(f).view(*f.shape[:-1], cfg.chunk, ACT_DIM)[:, :, 0]  # (B, T, 7) executed step
                rb = torch.where(bound > 0, bound * torch.tanh(r / bound.clamp_min(1e-6)), r)
                m = mask[:, :, 0] & b["state_avail"]  # valid steps with a plan
                base = b["latent"].view(*r.shape[:2], cfg.chunk, ACT_DIM)[:, :, 0]
                rb_all.append((rb[m][:, :6] / bound[:6]).abs())
                rg_all.append((r[m][:, 6] * std[6]).abs())
                l1_s.append(((base + rb)[m] - target[:, :, 0][m]).abs() * std)
                l1_p.append((base[m] - target[:, :, 0][m]).abs() * std)
            u = torch.cat(rb_all)  # |r_b| / bound, (N, 6)
            raw = u * 0.5  # bound is 0.5 raw on every arm dim
            row = {"n_steps": len(u), "any_arm_dim_at_95": (u >= 0.95).any(1).float().mean().item(),
                   "grip_abs_mean": torch.cat(rg_all).mean().item(),
                   "l1_student": torch.cat(l1_s).mean().item(), "l1_plan_only": torch.cat(l1_p).mean().item()}
            for g, sl in GROUPS.items():
                row[f"{g}_frac_at_95"] = (u[:, sl] >= 0.95).float().mean().item()
                row[f"{g}_frac_at_50"] = (u[:, sl] >= 0.5).float().mean().item()
                row[f"{g}_abs_mean"] = raw[:, sl].mean().item()
                row[f"{g}_abs_p95"] = raw[:, sl].flatten().quantile(0.95).item()
            out[f"k{k}"] = row
    finally:
        tr.state_source = orig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rows = []
    for su in SUITES:
        for size in SIZES:
            for sd in SEEDS:
                r = {"suite": su, "size": size, "seed": sd,
                     **residuals(RUNS / su / f"z-teacher-h40-track_{size}_s{sd}", su)}
                rows.append(r)
                print(su, size, sd, " ".join(f"k{k}: pos95 {r[f'k{k}']['pos_frac_at_95']:.3f} "
                                             f"rot95 {r[f'k{k}']['rot_frac_at_95']:.3f}" for k in KS), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
