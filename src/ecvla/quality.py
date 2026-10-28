"""Per-demo quality flags for a LIBERO task file (EXP-0001 A0), written as a sidecar next to the built dataset.

- first_success_step: first recorded state that satisfies the task's success predicate (-1 if none). Some original
  demos end before the predicate holds (e.g. the bottle still in the gripper), so training can filter on this.
- step_err: per-step replay error vs. the next recorded state (hybrid replay), with quaternions compared up to sign
  (q and -q are the same rotation) so only real divergence shows.
- unstable: steps where step_err exceeds UNSTABLE_THRESHOLD, e.g. contact penetration making an object jump;
  torque/F/T logged on those steps comes from an unphysical step.

Usage: python -m ecvla.quality <task.hdf5> <out.h5>
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from ecvla.libero_env import bddl_path, make_env

UNSTABLE_THRESHOLD = 0.05  # metres / radians / quaternion units, on any qpos entry


def _quat_starts(model) -> list[int]:
    return [int(model.jnt_qposadr[j]) + 3 for j in range(model.njnt) if model.jnt_type[j] == 0]  # free joints


def qpos_err(a: np.ndarray, b: np.ndarray, quat_starts: list[int]) -> np.ndarray:
    e = np.abs(a - b)
    for i in quat_starts:
        e[i : i + 4] = np.minimum(e[i : i + 4], np.abs(a[i : i + 4] + b[i : i + 4]))
    return e


def demo_quality(env, states: np.ndarray, actions: np.ndarray, quat_starts: list[int]) -> dict:
    env.reset()
    nq = env.sim.model.nq
    success = []
    for s in states:
        env.regenerate_obs_from_state(s)
        success.append(env.check_success())
    errs = []
    for t in range(len(actions) - 1):
        env.reset() if t and t % 500 == 0 else None  # stay below the env horizon on long demos
        env.regenerate_obs_from_state(states[t])
        env.step(actions[t])
        errs.append(qpos_err(env.get_sim_state()[1 : 1 + nq], states[t + 1][1 : 1 + nq], quat_starts).max())
    errs = np.asarray(errs, dtype=np.float32)
    return {
        "first_success_step": success.index(True) if any(success) else -1,
        "success_at_end": bool(success[-1]),
        "step_err": errs,
        "unstable": errs > UNSTABLE_THRESHOLD,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"skip: {out} exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    src = h5py.File(a.src)
    env = make_env(bddl_path(src["data"].attrs["bddl_file_name"]), cameras=False)
    qs = _quat_starts(env.sim.model._model)
    n_ok = n_unstable = 0
    with h5py.File(tmp, "w") as f:
        f.attrs.update(source_file=os.path.basename(a.src), unstable_threshold=UNSTABLE_THRESHOLD)
        for name in sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1])):
            d = src["data"][name]
            q = demo_quality(env, d["states"][()], d["actions"][()], qs)
            g = f.create_group(name)
            g.attrs.update(first_success_step=q["first_success_step"], success_at_end=q["success_at_end"])
            g.create_dataset("step_err", data=q["step_err"])
            g.create_dataset("unstable", data=q["unstable"])
            n_ok += q["first_success_step"] >= 0
            n_unstable += int(q["unstable"].any())
        f.attrs.update(demos_reaching_success=n_ok, demos_with_unstable_steps=n_unstable)
    os.replace(tmp, out)
    print("done", out, f"reaching_success={n_ok} with_unstable_steps={n_unstable}")


if __name__ == "__main__":
    main()
