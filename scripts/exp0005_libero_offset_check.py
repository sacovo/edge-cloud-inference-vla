"""EXP-0005 stage 0: which sim state do the LIBERO HDF5 observations belong to (proprio and images)?

For a few demos, replays `states[t + off]` (off = -1, 0, +1), renders both cameras at 128 px (the HDF5 resolution, same
OpenGL convention, no flip) and compares with the stored `agentview_rgb` / `eye_in_hand_rgb` and proprio at t.
The slow-only check found proprio matches at off = +1; this tells whether the images share that offset.

Runs in the VLM env with the project on the path (needs EGL rendering, i.e. a GPU node):
  PYTHONPATH=src uv run --project vlm python scripts/exp0005_libero_offset_check.py --out <json>
"""

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np

from ecvla.libero_env import make_env

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
SUITES = ("libero_spatial", "libero_object")
OFFSETS = (-1, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-per-suite", type=int, default=2)
    a = ap.parse_args()
    from libero.libero import get_libero_path

    rows = []
    for su in SUITES:
        for f in sorted(glob.glob(str(S / "raw" / su / "*.hdf5")))[: a.files_per_suite]:
            with h5py.File(f, "r") as h:
                bddl = os.path.join(get_libero_path("bddl_files"), su, Path(f).stem.removesuffix("_demo") + ".bddl")
                env = make_env(bddl, image_size=128, cameras=True)
                d = h["data/demo_0"]
                T = len(d["states"])
                for t in (1, T // 2, T - 2):
                    row = {"file": Path(f).name, "t": t}
                    for off in OFFSETS:
                        obs = env.set_init_state(d["states"][t + off])
                        pix = [float(np.abs(obs[sim].astype(np.int16) - d["obs"][rec][t].astype(np.int16)).mean())
                               for sim, rec in (("agentview_image", "agentview_rgb"),
                                                ("robot0_eye_in_hand_image", "eye_in_hand_rgb"))]
                        prop = float(np.abs(obs["robot0_eef_pos"] - d["obs/ee_pos"][t]).max())
                        row[f"off{off:+d}"] = {"agentview_mad": pix[0], "wrist_mad": pix[1], "eef_pos_max": prop}
                    rows.append(row)
                    print(row, flush=True)
                env.close()
    summary = {f"off{off:+d}": {k: float(np.mean([r[f"off{off:+d}"][k] for r in rows]))
                                for k in ("agentview_mad", "wrist_mad", "eef_pos_max")} for off in OFFSETS}
    print("summary (mean over frames; image MAD in 0-255 pixel units):", json.dumps(summary, indent=1))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
