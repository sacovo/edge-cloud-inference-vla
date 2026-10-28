"""EXP-0001 A0: check that LIBERO demos can be replayed to regenerate multi-rate sensors and ground-truth state.

Usage: uv run python scripts/a0_replay_check.py <demo.hdf5> [--episodes N] [--out results.json]
"""

import argparse
import json

import h5py
import numpy as np

from ecvla.libero_env import bddl_path, make_env
from ecvla.replay import replay_hybrid, replay_open_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()

    f = h5py.File(args.hdf5)
    env = make_env(bddl_path(f["data"].attrs["bddl_file_name"]))
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.episodes]
    rows = []
    for name in demos:
        d = f["data"][name]
        states, actions = d["states"][()], d["actions"][()]
        h = replay_hybrid(env, states, actions)
        o = replay_open_loop(env, states, actions)
        s = h["substeps"]
        row = {
            "demo": name,
            "steps": len(actions),
            "init_state_matches_states0": bool(np.allclose(d.attrs["init_state"], states[0])) if "init_state" in d.attrs else None,
            "hybrid_step_qpos_err_max": float(h["step_qpos_err"].max()),
            "hybrid_step_qpos_err_median": float(np.median(h["step_qpos_err"])),
            "substeps_per_step": sorted(set(h["substeps_per_step"].tolist())),
            "substep_dt": float(np.median(np.diff(s["time"]))),
            "joint_torque_absmax": float(np.abs(s["joint_torque"]).max()),
            "ee_force_absmax": float(np.abs(s["ee_force"]).max()),
            "ee_force_nonzero_frac": float((np.linalg.norm(s["ee_force"], axis=1) > 1e-6).mean()),
            "n_objects": len(h["objects"]),
            "recorded_success_state": h["success_at_end"],
            "open_loop_success": o["success"],
            "open_loop_success_step": o["success_step"],
            "open_loop_qpos_drift_max": float(o["qpos_drift"].max()),
        }
        rows.append(row)
        print(json.dumps(row))
    summary = {
        "file": args.hdf5,
        "episodes": len(rows),
        "open_loop_success_rate": float(np.mean([r["open_loop_success"] for r in rows])),
        "hybrid_step_qpos_err_max": max(r["hybrid_step_qpos_err_max"] for r in rows),
        "objects": h["objects"],
    }
    print("SUMMARY", json.dumps(summary))
    if args.out:
        json.dump({"summary": summary, "episodes": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
