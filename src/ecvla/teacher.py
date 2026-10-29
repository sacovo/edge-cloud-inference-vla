"""Slow "teacher" for EXP-0001 A1-teacher (H4): a trained large fast policy whose output becomes the latent z.

The teacher is a trained blind `task-id` policy (the 10M GRU of the same suite and seed). Its z is either its
predicted action chunk (CHUNK x 7, in its normalized action units, flattened) or its core features (d). The teacher
sees every observation (it is recurrent); the delay model only limits when its output reaches the fast policy.

Precompute on demo frames (training latents), one .npz per prepared task file, rows aligned with it:
    python -m ecvla.teacher <teacher_run_dir> <prepared_suite_dir> <out_dir>
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from ecvla.policy import FastPolicy, PolicyConfig
from ecvla.prepare import load
from ecvla.train import task_key

KINDS = ("chunk", "hidden")


class Teacher:
    def __init__(self, run_dir, device="cpu"):
        ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
        info = ckpt["info"]
        assert info["condition"] == "task-id" and info["vision"] == "none", "teacher must be a blind task-id policy"
        self.model = FastPolicy(PolicyConfig(**info["policy"])).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.stats = {k: (np.asarray(m, np.float32), np.asarray(s, np.float32)) for k, (m, s) in ckpt["stats"].items()}
        self.tasks, self.device, self.run_dir = ckpt["tasks"], device, str(run_dir)
        self.d = self.model.cfg.d

    def inputs(self, proprio, torque, ft, fast_valid, task: str) -> dict:
        """(T, ...) arrays of one episode -> a (1, T, ...) batch."""
        n = lambda k, x: (x - self.stats[k][0]) / self.stats[k][1]  # noqa: E731
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)[None]  # noqa: E731
        return {"proprio": t(n("proprio", proprio)), "torque": t(n("torque", torque)), "ft": t(n("ft", ft)),
                "fast_valid": t(fast_valid), "task": torch.tensor([self.tasks.index(task)], device=self.device)}

    @torch.no_grad()
    def latents(self, b: dict, state=None) -> tuple[dict, object]:
        """Both z kinds for every step of b: {"chunk": (T, CHUNK*7), "hidden": (T, d)}, plus the core state."""
        f, state = self.model.features(b, state)
        chunk = self.model.act(f)[0].flatten(-2)
        return {"chunk": chunk.cpu().numpy(), "hidden": f[0].cpu().numpy()}, state


class OnlineTeacher:
    """The teacher in the closed loop: one observation per control step, z of every step kept for the delay model."""

    def __init__(self, teacher: Teacher, task: str, kind: str):
        assert kind in KINDS
        self.teacher, self.task, self.kind = teacher, task, kind
        self.state, self.history = None, []

    def observe(self, obs: dict) -> None:
        x = {k: np.asarray(obs[k], np.float32)[None] for k in ("proprio", "torque", "ft")}
        b = self.teacher.inputs(x["proprio"], x["torque"], x["ft"], np.asarray([obs["fast_valid"]], np.float32), self.task)
        z, self.state = self.teacher.latents(b, self.state)
        self.history.append(z[self.kind][0])


def precompute(run_dir: str, suite_dir: str, out_dir: str) -> None:
    teacher = Teacher(run_dir, "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(suite_dir).glob("*.npz")):
        out = out_dir / f.name
        if out.exists():
            print(f"skip: {out} exists")
            continue
        arr, meta = load(f)
        starts = arr["ep_start"]
        parts = {k: [] for k in KINDS}
        for d in range(len(starts) - 1):
            sl = slice(starts[d], starts[d + 1])
            b = teacher.inputs(arr["proprio"][sl], arr["torque"][sl], arr["ft"][sl], arr["fast_valid"][sl], task_key(meta))
            z, _ = teacher.latents(b)
            for k in KINDS:
                parts[k].append(z[k])
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **{f"z_{k}": np.concatenate(v).astype(np.float16) for k, v in parts.items()},
                 teacher=np.asarray(str(run_dir)))
        os.replace(tmp, out)
        print("done", out, starts[-1], "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("suite_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    precompute(a.run_dir, a.suite_dir, a.out_dir)


if __name__ == "__main__":
    main()
