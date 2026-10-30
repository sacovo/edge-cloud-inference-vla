"""Train one EXP-0001 fast policy (A1: conditions none / task-id / oracle-delayed / oracle-fresh).

Data: prepare.py outputs (one .npz per task, plus vision.py features for `--vision full`). Demos whose recorded
states never reach the success predicate are dropped; the last `--val-per-task` demos of each task are held out for
the validation loss (closed-loop success is measured by evaluate.py on LIBERO's own initial states).

Training runs on whole episodes (the core starts from zero at t=0, as in closed loop). For `oracle-delayed`, every
episode in every batch draws an update period k from K_TRAIN and a delay of k-1..k+1 steps (README §3: "same k set
as evaluation, plus jitter of ±1"); the policy sees the newest object state that has arrived, as in latency.py.

Usage: python -m ecvla.train --data <prepared_dir> --suites libero_spatial --condition oracle-fresh --size 1000000
           --seed 0 --out <run_dir> [--vision full --vision-dir <dir>] [--core gru|tf] [--head l1|flow]
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ecvla.latency import DelayModel, schedule
from ecvla.policy import ACT_DIM, FastPolicy, config_dict, config_for_size, plan_window
from ecvla.prepare import load
from ecvla.vision import FEATURE_DIM

CONDITIONS = {"none": "none", "task-id": "task", "oracle-delayed": "state", "oracle-fresh": "state",
              "z-teacher": "latent"}
K_TRAIN = {"oracle-delayed": (1, 10), "z-teacher": (1, 4, 10, 20)}  # Phase A axis D; A1-teacher (H4)
CONTROL_HZ = 20.0
NORMALIZED = ("proprio", "torque", "ft", "obj_pose", "vision", "latent", "actions")


def task_key(meta: dict) -> str:
    return Path(meta["bddl_file_name"]).stem


class EpisodeData:
    """All demos of the selected suites as flat per-step arrays (rows), with per-episode index ranges."""

    def __init__(self, data_dir, suites, vision_dir=None, val_per_task=2, latent_dir=None, latent_kind="chunk"):
        files = sorted(f for s in suites for f in (Path(data_dir) / s).glob("*.npz"))
        assert files, f"no prepared files for {suites} in {data_dir}"
        parts, self.tasks, self.objects, self.fixtures = [], [], {}, {}
        names = set()
        for f in files:
            arr, meta = load(f)
            if vision_dir is not None:
                arr["vision"] = np.load(Path(vision_dir) / meta["suite"] / f.name)["vision"].astype(np.float32)
                assert len(arr["vision"]) == len(arr["actions"]), f
            if latent_dir is not None:
                arr["latent"] = np.load(Path(latent_dir) / meta["suite"] / f.name)[f"z_{latent_kind}"].astype(np.float32)
                assert len(arr["latent"]) == len(arr["actions"]), f
            parts.append((arr, meta))
            self.tasks.append(task_key(meta))
            self.objects[task_key(meta)] = meta["objects"]
            self.fixtures[task_key(meta)] = meta.get("fixtures", [])
            if "fix_state" not in arr:  # prepared before fixture state existed (A1-arch, v1)
                arr["fix_state"] = np.zeros((len(arr["actions"]), 0, 2), np.float32)
            names.update(meta["objects"])
            names.update(self.fixtures[task_key(meta)])
        self.names = sorted(names)
        self.max_obj = max(len(o) for o in self.objects.values())
        self.max_fix = max(len(o) for o in self.fixtures.values())
        rows = {k: [] for k in parts[0][0] if k not in ("ep_start", "first_success_step")}
        self.episodes = []  # (start_row, length, task_idx, is_val)
        offset = 0
        for ti, (arr, meta) in enumerate(parts):
            n_obj = len(meta["objects"])
            pad = self.max_obj - n_obj
            arr["obj_pose"] = np.pad(arr["obj_pose"], ((0, 0), (0, pad), (0, 0)))
            arr["contact"] = np.pad(arr["contact"], ((0, 0), (0, pad)))
            arr["fix_state"] = np.pad(arr["fix_state"], ((0, 0), (0, self.max_fix - arr["fix_state"].shape[1]), (0, 0)))
            for k in rows:
                rows[k].append(arr[k])
            starts = arr["ep_start"]
            kept = [d for d in range(len(starts) - 1) if arr["first_success_step"][d] >= 0]
            val = set(kept[-val_per_task:]) if val_per_task else set()
            for d in kept:
                self.episodes.append((offset + starts[d], starts[d + 1] - starts[d], ti, d in val))
            offset += starts[-1]
        self.rows = {k: np.concatenate(v).astype(np.float32) for k, v in rows.items()}
        self.row_task = np.concatenate([np.full(p[0]["ep_start"][-1], ti) for ti, p in enumerate(parts)])
        self.obj_name = np.zeros((len(self.tasks), self.max_obj), np.int64)
        self.obj_mask = np.zeros((len(self.tasks), self.max_obj), bool)
        for ti, t in enumerate(self.tasks):
            objs = self.objects[t]
            self.obj_name[ti, : len(objs)] = [self.names.index(o) for o in objs]
            self.obj_mask[ti, : len(objs)] = True
        self.fix_name = np.zeros((len(self.tasks), self.max_fix), np.int64)
        self.fix_mask = np.zeros((len(self.tasks), self.max_fix), bool)
        for ti, t in enumerate(self.tasks):
            fx = self.fixtures[t]
            self.fix_name[ti, : len(fx)] = [self.names.index(o) for o in fx]
            self.fix_mask[ti, : len(fx)] = True

    def split(self, val: bool) -> list:
        return [e for e in self.episodes if e[3] == val]

    def norm_stats(self) -> dict:
        train_rows = np.concatenate([np.arange(s, s + n) for s, n, _, v in self.episodes if not v])
        stats = {}
        for k in NORMALIZED:
            if k not in self.rows:
                continue
            x = self.rows[k][train_rows]
            if k == "obj_pose":
                x = x[self.obj_mask[self.row_task[train_rows]]]  # real object slots only
            x = x.reshape(-1, x.shape[-1])
            stats[k] = (x.mean(0).tolist(), np.maximum(x.std(0), 1e-3).tolist())
        return stats


def state_source(T: int, condition: str, rng: np.random.Generator, k: int | None = None, train: bool = True):
    """(src, age_seconds) per step for the delayed conditions; src = -1 before the first update arrives."""
    if condition == "oracle-fresh":
        src = np.arange(T)
    else:
        if k is None:
            k = int(rng.choice(K_TRAIN[condition]))
        delay = DelayModel(base=k - 1, jitter=2) if train else DelayModel(base=k)
        src = schedule(T, period=k, delay=delay, rng=rng).src
    age = np.where(src >= 0, np.arange(T) - src, 0) / CONTROL_HZ
    return src, age.astype(np.float32)


class Batcher:
    def __init__(self, data: EpisodeData, stats: dict, condition: str, device, chunk: int, track: bool = False):
        """track (EXP-0003): z is the plan window due at each step's age, in the plan's own (normalized action) units."""
        self.data, self.condition, self.device, self.chunk, self.track = data, condition, device, chunk, track
        self.rows = {}
        for k, x in data.rows.items():
            if k in stats:
                m, s = (np.asarray(v, np.float32) for v in stats[k])
                x = (x - m) / s
            self.rows[k] = torch.from_numpy(x).to(device)
        self.obj_name = torch.from_numpy(data.obj_name).to(device)
        self.obj_mask = torch.from_numpy(data.obj_mask).to(device)
        self.fix_name = torch.from_numpy(data.fix_name).to(device)
        self.fix_mask = torch.from_numpy(data.fix_mask).to(device)

    def __call__(self, episodes: list, rng: np.random.Generator) -> tuple[dict, torch.Tensor, torch.Tensor]:
        B, T = len(episodes), max(e[1] for e in episodes)
        t = np.arange(T)
        idx = np.zeros((B, T), np.int64)
        step_mask = np.zeros((B, T), bool)
        src_idx = np.zeros((B, T), np.int64)
        avail = np.zeros((B, T), bool)
        age = np.zeros((B, T), np.float32)
        tgt_idx = np.zeros((B, T, self.chunk), np.int64)
        tgt_mask = np.zeros((B, T, self.chunk), bool)
        for i, (start, n, _, _) in enumerate(episodes):
            idx[i] = start + np.minimum(t, n - 1)
            step_mask[i, :n] = True
            j = t[:, None] + np.arange(self.chunk)[None]
            tgt_idx[i] = start + np.minimum(j, n - 1)
            tgt_mask[i] = (j < n) & step_mask[i, :, None]
            if self.condition != "none" and self.condition != "task-id":
                src, a = state_source(n, self.condition, rng)
                src_idx[i, :n], avail[i, :n], age[i, :n] = start + np.maximum(src, 0), src >= 0, a
        dev = self.device
        g = lambda k, ix: self.rows[k][torch.from_numpy(ix).to(dev)]  # noqa: E731
        task = torch.tensor([e[2] for e in episodes], device=dev)
        b = {k: g(k, idx) for k in ("proprio", "torque", "ft", "fast_valid")}
        if "vision" in self.rows:
            b["vision"] = g("vision", idx)
        b.update(task=task, obj_name=self.obj_name[task], obj_mask=self.obj_mask[task])
        b.update(obj_pose=g("obj_pose", src_idx), contact=g("contact", src_idx))
        b.update(fix_state=g("fix_state", src_idx), fix_name=self.fix_name[task], fix_mask=self.fix_mask[task])
        if "latent" in self.rows:
            b["latent"] = g("latent", src_idx)
            if self.track:
                age_steps = torch.from_numpy(np.rint(age * CONTROL_HZ).astype(np.int64)).to(dev)
                b["latent"] = plan_window(b["latent"], age_steps, self.chunk)
        b["state_avail"] = torch.from_numpy(avail).to(dev)
        b["age"] = torch.from_numpy(age).to(dev)
        target = g("actions", tgt_idx)
        return b, target, torch.from_numpy(tgt_mask).to(dev)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None].to(x.dtype)
    return (x * m).sum() / (m.sum() * x.shape[-1])


@torch.no_grad()
def val_first_action_l1(model, batcher, episodes, rng, std) -> float:
    """L1 (in action units) of the first action of each chunk on held-out demos."""
    model.eval()
    tot, n = 0.0, 0
    gen = torch.Generator(device=batcher.device).manual_seed(0)
    for i in range(0, len(episodes), 16):
        b, target, mask = batcher(episodes[i : i + 16], rng)
        f, _ = model.features(b)
        err = ((model.act(f, gen, b)[:, :, 0] - target[:, :, 0]).abs() * std).sum(-1)
        tot += float((err * mask[:, :, 0]).sum())
        n += int(mask[:, :, 0].sum()) * ACT_DIM
    model.train()
    return tot / n


def train(a) -> dict:
    out = Path(a.out)
    if (out / "model.pt").exists():
        print(f"skip: {out} has model.pt")
        return {}
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.require_gpu and device.type != "cuda":
        raise SystemExit("--require-gpu: CUDA unavailable (driver/torch mismatch?)")
    latent = a.condition == "z-teacher"
    assert not latent or a.latent_dir, "z-teacher needs --latent-dir (teacher.py output)"
    data = EpisodeData(a.data, a.suites, a.vision_dir if a.vision == "full" else None, a.val_per_task,
                       a.latent_dir if latent else None, a.latent_kind)
    stats = data.norm_stats()
    track = {}
    if a.track:  # the plan is already in normalized action units; bound the arm dims, not the gripper
        assert latent and a.latent_kind == "chunk", "--track needs a z-teacher chunk latent"
        stats.pop("latent")
        std = np.asarray(stats["actions"][1])
        track = dict(track=True, track_bound=tuple([a.track_bound / x for x in std[:6]] + [0.0]))
    cfg = config_for_size(
        a.size, cond=CONDITIONS[a.condition], n_tasks=len(data.tasks), n_names=len(data.names),
        vision_dim=FEATURE_DIM if a.vision == "full" else 0, age_input=a.age_input, core=a.core, head=a.head,
        fixtures=CONDITIONS[a.condition] == "state" and data.max_fix > 0,
        latent_dim=(a.chunk * ACT_DIM if a.track else data.rows["latent"].shape[-1]) if latent else 0, chunk=a.chunk,
        **track,
    )
    model = FastPolicy(cfg).to(device)
    batcher = Batcher(data, stats, a.condition, device, cfg.chunk, cfg.track)
    train_eps, val_eps = data.split(False), data.split(True)
    act_std = torch.tensor(stats["actions"][1], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    warmup = max(1, min(500, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))
    )
    info = {
        "condition": a.condition, "vision": a.vision, "size_target": a.size, "seed": a.seed, "suites": a.suites,
        "policy": config_dict(cfg), "head_params": model.head_params(), "vision_params": model.vision_params(),
        "train_episodes": len(train_eps), "val_episodes": len(val_eps), "git_sha": git_sha(), "device": str(device), "args": vars(a),
    }
    print(json.dumps(info))
    log = open(out / "train_log.jsonl", "a")
    amp = device.type == "cuda"
    t0 = time.time()
    for step in range(1, a.steps + 1):
        eps = [train_eps[i] for i in rng.integers(0, len(train_eps), a.batch)]
        b, target, mask = batcher(eps, rng)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            f, _ = model.features(b)
            loss = masked_mean(model.loss(f.float(), target, b), mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": loss.item(), "grad_norm": gn.item(), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1)}
            if val_eps and (step % (a.log_every * 10) == 0 or step == a.steps):
                rec["val_l1"] = val_first_action_l1(model, batcher, val_eps, np.random.default_rng(0), act_std)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    info["final"] = rec
    torch.save({"state_dict": model.state_dict(), "info": info, "stats": stats, "tasks": data.tasks,
                "objects": data.objects, "names": data.names, "max_obj": data.max_obj,
                "fixtures": data.fixtures, "max_fix": data.max_fix}, out / "model.pt")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    return info


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--condition", choices=list(CONDITIONS), required=True)
    ap.add_argument("--size", type=int, required=True, help="target head params")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vision", choices=["none", "full"], default="none")
    ap.add_argument("--vision-dir")
    ap.add_argument("--core", choices=["gru", "tf"], default="gru")
    ap.add_argument("--head", choices=["l1", "flow"], default="l1")
    ap.add_argument("--age-input", action="store_true")
    ap.add_argument("--chunk", type=int, default=8, help="predicted action-chunk length (EXP-0002 teachers: 40)")
    ap.add_argument("--latent-dir", help="z-teacher: teacher.py output root (<dir>/<suite>/<task>.npz)")
    ap.add_argument("--latent-kind", choices=["chunk", "hidden"], default="chunk")
    ap.add_argument("--teacher", help="z-teacher: teacher run dir (recorded for closed-loop eval)")
    ap.add_argument("--track", action="store_true",
                    help="EXP-0003: age-aligned plan window as z, output = plan + bounded per-step residual")
    ap.add_argument("--track-bound", type=float, default=0.5, help="residual bound on the arm dims, raw action units")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32, help="episodes per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-per-task", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device")
    ap.add_argument("--require-gpu", action="store_true", help="fail instead of silently training on CPU")
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
