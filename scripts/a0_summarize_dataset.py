"""Summarize a built Phase A dataset directory (<dir>/<suite>/<task>.h5): counts, replay quality, size.

Usage: uv run python scripts/a0_summarize_dataset.py <dataset_dir> [--out summary.json]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    suites = {}
    for f in sorted(Path(a.dir).glob("*/*.h5")):
        with h5py.File(f) as h:
            demos = [h[k] for k in h.keys()]
            row = {
                "task": f.stem,
                "instruction": h.attrs["language_instruction"],
                "demos": len(demos),
                "steps": int(sum(d.attrs["n_steps"] for d in demos)),
                "open_loop_success": int(sum(bool(d.attrs["open_loop_success"]) for d in demos)),
                "recorded_success_state": int(sum(bool(d.attrs["recorded_success_state"]) for d in demos)),
                "max_step_qpos_err": float(max(d["step_qpos_err"][()].max() for d in demos)),
                "n_objects": len(json.loads(demos[0].attrs["objects"])),
                "bytes": f.stat().st_size,
            }
        suites.setdefault(f.parent.name, []).append(row)
    summary = {}
    for s, rows in suites.items():
        summary[s] = {
            "tasks": len(rows),
            "demos": sum(r["demos"] for r in rows),
            "steps": sum(r["steps"] for r in rows),
            "steps_per_demo_mean": round(sum(r["steps"] for r in rows) / sum(r["demos"] for r in rows), 1),
            "open_loop_success_rate": round(sum(r["open_loop_success"] for r in rows) / sum(r["demos"] for r in rows), 4),
            "recorded_success_rate": round(sum(r["recorded_success_state"] for r in rows) / sum(r["demos"] for r in rows), 4),
            "max_step_qpos_err": round(max(r["max_step_qpos_err"] for r in rows), 4),
            "objects_per_task": sorted({r["n_objects"] for r in rows}),
            "gb": round(sum(r["bytes"] for r in rows) / 1e9, 2),
            "worst_open_loop_tasks": sorted(((r["open_loop_success"], r["demos"], r["task"]) for r in rows))[:3],
        }
    print(json.dumps(summary, indent=2))
    if a.out:
        json.dump({"suites": summary, "tasks": suites}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
