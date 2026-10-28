"""Build the EXP-0001 Phase A dataset from one original LIBERO task hdf5.

For every demo: hybrid replay (see replay.py), writing
- substeps/*      500 Hz float32: time, joint_torque, joint_pos, joint_vel, gripper_qpos, ee_force, ee_torque
- object_pose     20 Hz float32 (T, 7 * n_objects): [pos(3), quat(4)] per object, order in attrs["objects"]
- actions         20 Hz float32 (T, 7), copied from the source
- sim_state       20 Hz float64 (T, nq + nv + 1), copied from the source
- images/<cam>    JPEG-encoded frames at --image-size, rendered from sim_state[t] (aligned with source obs[t])
- step_qpos_err   per-step replay error vs. the next recorded state
Source 128 px images are not copied; attrs["source_file"] + demo name reference them.

The output is written to <out>.tmp and renamed when complete, so an existing output means a finished task.

Usage: python -m ecvla.build_dataset <task.hdf5> <out.h5> [--image-size 224] [--max-demos N]
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from ecvla.libero_env import bddl_path, make_env
from ecvla.replay import CAMERAS, replay_hybrid, replay_open_loop

JPEG_QUALITY = 95


def encode_jpeg(frames: np.ndarray) -> list[np.ndarray]:
    out = []
    for fr in frames:
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, format="JPEG", quality=JPEG_QUALITY)
        out.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
    return out


def decode_jpeg(data: np.ndarray) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data.tobytes())))


def build(src_path: str, out_path: str, image_size: int = 224, max_demos: int | None = None) -> dict:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"skip: {out_path} exists")
        return {"skipped": True}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = h5py.File(src_path)
    sattrs = src["data"].attrs
    env = make_env(bddl_path(sattrs["bddl_file_name"]), image_size=image_size)
    demos = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))[:max_demos]
    vlen = h5py.vlen_dtype(np.uint8)
    stats = {"demos": 0, "steps": 0, "open_loop_success": 0, "max_step_qpos_err": 0.0}
    t0 = time.time()
    with h5py.File(tmp, "w") as out:
        out.attrs.update(
            source_file=os.path.basename(src_path),
            bddl_file_name=sattrs["bddl_file_name"],
            language_instruction=env.language_instruction,
            image_size=image_size,
            image_convention=sattrs.get("macros_image_convention", "opengl"),
            cameras=json.dumps(CAMERAS),
            control_freq=env.env.control_freq,
            sim_timestep=float(env.sim.model.opt.timestep),
        )
        for name in demos:
            d = src["data"][name]
            states, actions = d["states"][()], d["actions"][()]
            h = replay_hybrid(env, states, actions, render=True)
            ol = replay_open_loop(env, states, actions)
            g = out.create_group(name)
            for k, v in h["substeps"].items():
                g.create_dataset(f"substeps/{k}", data=v.astype(np.float32), compression="gzip")
            g.create_dataset("object_pose", data=h["object_pose"].astype(np.float32), compression="gzip")
            g.create_dataset("actions", data=actions.astype(np.float32))
            g.create_dataset("sim_state", data=states, compression="gzip")
            g.create_dataset("step_qpos_err", data=h["step_qpos_err"].astype(np.float32))
            for cam, frames in h["images"].items():
                enc = encode_jpeg(frames)
                ds = g.create_dataset(f"images/{cam}", shape=(len(enc),), dtype=vlen)
                for i, e in enumerate(enc):
                    ds[i] = e
            g.attrs.update(
                objects=json.dumps(h["objects"]),
                n_steps=len(actions),
                recorded_success_state=h["success_at_end"],
                open_loop_success=ol["success"],
            )
            stats["demos"] += 1
            stats["steps"] += len(actions)
            stats["open_loop_success"] += int(ol["success"])
            stats["max_step_qpos_err"] = max(stats["max_step_qpos_err"], float(h["step_qpos_err"].max()))
        out.attrs["stats"] = json.dumps(stats)
    env.close()
    os.replace(tmp, out_path)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes"] = out_path.stat().st_size
    print("done", out_path, json.dumps(stats))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-demos", type=int)
    a = ap.parse_args()
    build(a.src, a.out, a.image_size, a.max_demos)


if __name__ == "__main__":
    main()
