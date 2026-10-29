"""Aggregate A1 closed-loop results: <runs_dir>/<suite>/<run>/eval/<suite>_t<task>_k<k>.json -> one row per run and k.

Success rate is the mean over tasks (50 episodes each); incomplete runs (fewer than 10 tasks) are flagged.
Usage: python3 scripts/a1_summarize.py <runs_dir> [--out summary.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(Path(a.runs_dir).glob("*/*/eval/*.json")):
        r = json.loads(f.read_text())
        groups[(f.parts[-4], f.parts[-3], r["k"])].append(r)
    rows = []
    for (suite, run, k), rs in sorted(groups.items()):
        info = rs[0]["info"]
        rows.append({
            "suite": suite, "run": run, "k": k, "condition": info["condition"], "vision": info["vision"],
            "size_target": info["size_target"], "head_params": info["head_params"],
            "arch": f'{info["policy"]["core"]}-{info["policy"]["head"]}', "seed": info["seed"],
            "tasks": len(rs), "success_rate": round(sum(r["success_rate"] for r in rs) / len(rs), 4),
            "per_task": {r["task_id"]: r["success_rate"] for r in sorted(rs, key=lambda r: r["task_id"])},
            "val_l1": info.get("final", {}).get("val_l1"),
        })
    print(f"{'suite':15} {'run':45} {'k':>3} {'tasks':>5} {'SR':>6} {'val_l1':>7}")
    for r in rows:
        flag = "" if r["tasks"] == 10 else "  (incomplete)"
        print(f"{r['suite']:15} {r['run']:45} {str(r['k']):>3} {r['tasks']:>5} {r['success_rate']:6.3f} "
              f"{r['val_l1'] or float('nan'):7.4f}{flag}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
