"""Compact per-step training arrays for the EXP-0001 fast policy, one .npz per LIBERO task (A1).

Reads the Phase A dataset file (build_dataset.py) and its quality sidecar (quality.py), and adds gripper-object
contact flags (part of the oracle state; not in the dataset), computed by setting the sim to each recorded state.
Arrays are concatenated over demos; demo d spans rows ep_start[d]:ep_start[d+1]. See sensors.py for the inputs.

Usage: python -m ecvla.prepare <phaseA_task.h5> <quality_task.h5> <out.npz>
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from ecvla.libero_env import bddl_path, make_env
from ecvla.sensors import fast_windows, gripper_contacts, proprio_from_state


def demo_arrays(env, g, q, nq: int, objects: list[str]) -> dict:
    states = g["sim_state"][()]
    T = len(states)
    sub = {k: g[f"substeps/{k}"][()] for k in ("joint_torque", "ee_force", "ee_torque")}
    torque, ft = fast_windows(sub, T)
    fast_valid = np.ones(T, np.float32)
    fast_valid[0] = 0
    fast_valid[1:] = ~q["unstable"][()]  # unstable[t]: transition t -> t+1, whose substeps feed observation t+1
    contact = np.zeros((T, len(objects)), np.float32)
    env.reset()
    for t, s in enumerate(states):
        if t and t % 500 == 0:
            env.reset()  # stay below the env horizon on long demos
        env.regenerate_obs_from_state(s)
        contact[t] = gripper_contacts(env, objects)
    return {
        "proprio": proprio_from_state(states[:, 1 : 1 + nq], states[:, 1 + nq :]).astype(np.float32),
        "torque": torque,
        "ft": ft,
        "fast_valid": fast_valid,
        "obj_pose": g["object_pose"][()].reshape(T, len(objects), 7),
        "contact": contact,
        "actions": g["actions"][()],
    }


def prepare(data_path: str, quality_path: str, out_path: str, max_demos: int | None = None) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"skip: {out_path} exists")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path) as f, h5py.File(quality_path) as qf:
        env = make_env(bddl_path(f.attrs["bddl_file_name"]), cameras=False)
        nq = env.sim.model.nq
        demos = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:max_demos]
        objects = json.loads(f[demos[0]].attrs["objects"])
        parts, first_success = [], []
        for name in demos:
            assert json.loads(f[name].attrs["objects"]) == objects, name
            parts.append(demo_arrays(env, f[name], qf[name], nq, objects))
            first_success.append(int(qf[name].attrs["first_success_step"]))
        env.close()
        lengths = [len(p["actions"]) for p in parts]
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        meta = {
            "suite": Path(data_path).parent.name,
            "task": Path(data_path).stem,
            "bddl_file_name": str(f.attrs["bddl_file_name"]),
            "language_instruction": str(f.attrs["language_instruction"]),
            "objects": objects,
            "demos": demos,
        }
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **arrays,
        ep_start=np.r_[0, np.cumsum(lengths)],
        first_success_step=np.asarray(first_success),
        meta=np.asarray(json.dumps(meta)),
    )
    os.replace(tmp, out_path)
    print("done", out_path, f"demos={len(demos)} steps={sum(lengths)} contact_rate={arrays['contact'].mean():.3f}")


def load(path) -> tuple[dict, dict]:
    z = np.load(path)
    arrays = {k: z[k] for k in z.files if k != "meta"}
    return arrays, json.loads(str(z["meta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("quality")
    ap.add_argument("out")
    ap.add_argument("--max-demos", type=int)
    a = ap.parse_args()
    prepare(a.data, a.quality, a.out, a.max_demos)


if __name__ == "__main__":
    main()
