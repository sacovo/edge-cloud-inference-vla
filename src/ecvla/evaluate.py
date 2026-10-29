"""Closed-loop LIBERO evaluation of a trained EXP-0001 fast policy, one task per call.

Protocol: LIBERO's fixed initial states for the task (50, never edited), 10 no-op steps for objects to settle, then
the policy acts at 20 Hz until success or MAX_STEPS (the common OpenVLA/openpi limits per suite). Only the first
action of each predicted chunk is executed. Inputs come from sensors.OnlineSensors, which builds them the same way
as prepare.py. For `oracle-delayed`, object state is refreshed every k steps and arrives exactly k steps late.

Usage: python -m ecvla.evaluate <run_dir> --suite libero_spatial --task 0 --out <result.json> [--k 1]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ecvla.libero_env import make_env
from ecvla.policy import FastPolicy, PolicyConfig
from ecvla.replay import SubstepLogger, object_names
from ecvla.sensors import NOOP_ACTION, OnlineSensors
from ecvla.train import state_source, task_key

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SETTLE_STEPS = 10


class Runner:
    """Stateful policy for one episode: feed one observation per step, get one action."""

    def __init__(self, ckpt: dict, task: str, device="cpu"):
        info = ckpt["info"]
        self.cfg = PolicyConfig(**info["policy"])
        self.model = FastPolicy(self.cfg).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.condition = info["condition"]
        self.stats = {k: (np.asarray(m, np.float32), np.asarray(s, np.float32)) for k, (m, s) in ckpt["stats"].items()}
        self.device = device
        ti = ckpt["tasks"].index(task)
        objs = ckpt["objects"][task]
        S = ckpt["max_obj"]
        name = np.zeros(S, np.int64)
        name[: len(objs)] = [ckpt["names"].index(o) for o in objs]
        mask = np.zeros(S, bool)
        mask[: len(objs)] = True
        fx = ckpt.get("fixtures", {}).get(task, [])
        J = ckpt.get("max_fix", 0)
        fname = np.zeros(J, np.int64)
        fname[: len(fx)] = [ckpt["names"].index(o) for o in fx]
        fmask = np.zeros(J, bool)
        fmask[: len(fx)] = True
        t = lambda x: torch.as_tensor(x, device=device)[None]  # noqa: E731
        self.episode_inputs = {"task": t(ti), "obj_name": t(name), "obj_mask": t(mask), "fix_name": t(fname),
                               "fix_mask": t(fmask)}
        self.n_obj, self.S, self.n_fix, self.J = len(objs), S, len(fx), J

    def norm(self, k, x):
        m, s = self.stats[k]
        return (x - m) / s

    def reset(self, horizon: int, rng: np.random.Generator, k: int | None):
        self.state, self.history, self.t = None, [], 0
        if self.condition.startswith("oracle"):
            self.src, self.age = state_source(horizon, self.condition, rng, k=k, train=False)
        self.gen = torch.Generator(device=self.device).manual_seed(int(rng.integers(2**31)))

    @torch.no_grad()
    def act(self, obs: dict, vision: np.ndarray | None = None) -> np.ndarray:
        step = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)[None, None]  # noqa: E731
        b = dict(self.episode_inputs)
        b.update(proprio=step(self.norm("proprio", obs["proprio"])), torque=step(self.norm("torque", obs["torque"])),
                 ft=step(self.norm("ft", obs["ft"])), fast_valid=step(obs["fast_valid"]))
        if vision is not None:
            b["vision"] = step(self.norm("vision", vision))
        pose = np.zeros((self.S, 7), np.float32)
        contact = np.zeros(self.S, np.float32)
        fix = np.zeros((self.J, 2), np.float32)
        self.history.append((obs["obj_pose"], obs["contact"], obs.get("fix_state")))
        avail, age = False, 0.0
        if self.condition.startswith("oracle") and self.src[self.t] >= 0:
            p, c, f = self.history[self.src[self.t]]
            pose[: self.n_obj], contact[: self.n_obj] = p, c
            if self.n_fix:
                fix[: self.n_fix] = f
            avail, age = True, self.age[self.t]
        b.update(obj_pose=step(self.norm("obj_pose", pose)), contact=step(contact),
                 fix_state=step(fix), state_avail=torch.tensor([[avail]], device=self.device), age=step(age))
        f, self.state = self.model.features(b, self.state)
        chunk = self.model.act(f, self.gen)[0, 0].cpu().numpy()
        self.t += 1
        m, s = self.stats["actions"]
        return chunk[0] * s + m


def evaluate(run_dir: str, suite: str, task_id: int, out: str, k: int | None = None, episodes: int = 50,
             seed: int = 0) -> dict:
    out = Path(out)
    if out.exists():
        print(f"skip: {out} exists")
        return json.loads(out.read_text())
    from libero.libero import benchmark, get_libero_path

    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    bm = benchmark.get_benchmark_dict()[suite]()
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    key = Path(bddl).stem
    runner = Runner(ckpt, key)
    if runner.condition == "oracle-delayed":
        assert k is not None, "oracle-delayed needs --k"
    vision = ckpt["info"]["vision"] == "full"
    encoder = None
    if vision:
        from ecvla.vision import FrozenEncoder

        encoder = FrozenEncoder("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(bddl, image_size=224, cameras=vision)
    init_states = bm.get_task_init_states(task_id)
    horizon = MAX_STEPS[suite]
    rng = np.random.default_rng(seed)
    results, t0 = [], time.time()
    for ep in range(min(episodes, len(init_states))):
        env.reset()
        logger = SubstepLogger(env)  # after reset: a hard reset rebuilds the sim and drops the step hook
        obs = env.set_init_state(init_states[ep])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(NOOP_ACTION)
        objs = object_names(obs)
        assert objs == ckpt["objects"][key], (objs, ckpt["objects"][key])
        sensors = OnlineSensors(env, logger, objs)
        if runner.cfg.fixtures:  # models without fixture tokens (no fixture data) ignore the scene's fixtures
            assert sensors.fixtures == ckpt["fixtures"][key], (sensors.fixtures, ckpt["fixtures"][key])
        logger.take()
        runner.reset(horizon, rng, k)
        success, steps = False, 0
        for steps in range(1, horizon + 1):
            o = sensors.observe(obs)
            a = runner.act(o, encoder.encode_obs(obs) if encoder else None)
            obs, _, done, _ = env.step(np.clip(a, -1, 1))
            if done or env.check_success():
                success = True
                break
        logger.detach()
        results.append({"episode": ep, "success": success, "steps": steps})
        print(f"ep {ep} success={success} steps={steps}", flush=True)
    env.close()
    res = {
        "run": str(run_dir), "suite": suite, "task_id": task_id, "task": key, "condition": runner.condition,
        "k": k, "episodes": len(results), "success_rate": float(np.mean([r["success"] for r in results])),
        "per_episode": results, "seconds": round(time.time() - t0, 1), "info": ckpt["info"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done", out, res["success_rate"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    evaluate(a.run_dir, a.suite, a.task, a.out, a.k, a.episodes, a.seed)


if __name__ == "__main__":
    main()
