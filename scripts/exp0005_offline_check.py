"""EXP-0005 diagnosis: does the fine-tuned `pi0_fast` reproduce the LIBERO demo actions on training frames?

Separates "the model is poor" from "the closed loop is broken": on demo frames rendered exactly as in the mixture
(`exp0005_dtype_check.frames`), the greedy-decoded 1 s chunk is compared with the demo's next 20 actions, per action
dimension, in normalized units. Baselines on the same frames: the zero action (the quantile midpoint) and the
demo's current action repeated over the chunk (persistence). Also: gripper sign agreement (pos / rot / grip split).

  PYTHONPATH=src uv run --project vlm python scripts/exp0005_offline_check.py --ckpt <.../pretrained_model> \
    --mixture <mixture dir> --out <json> [--tasks 0 1 ...] [--stride 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp0005_dtype_check import frames  # noqa: E402
from exp0005_slow_only import load_policy, predict_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    mix = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    policy, pre, post, info = load_policy(a.ckpt, "float32")
    rows, failed = [], 0
    for suite, stem, t, batch, rec in frames(mix, ("libero_spatial", "libero_object"), a.tasks, a.stride):
        c = predict_chunk(policy, pre, post, batch)
        if c is None:
            failed += 1
            continue
        n = min(len(c), len(rec))
        rows.append({"suite": suite, "task": stem, "step": t,
                     "model": np.abs(c[:n] - rec[:n]).mean(0), "zero": np.abs(rec[:n]).mean(0),
                     "persist": np.abs(rec[:1] - rec[:n]).mean(0),
                     "grip_agree": float((np.sign(c[:n, 6]) == np.sign(rec[:n, 6])).mean())})
    out = {"ckpt": a.ckpt, "policy": info, "frames": len(rows), "decode_failed": failed}
    for suite in ("libero_spatial", "libero_object", None):
        rs = [r for r in rows if suite is None or r["suite"] == suite]
        key = suite or "pooled"
        out[key] = {k: np.mean([r[k] for r in rs], 0).round(4).tolist() for k in ("model", "zero", "persist")}
        out[key]["grip_agree"] = round(float(np.mean([r["grip_agree"] for r in rs])), 4)
        out[key]["frames"] = len(rs)
    print(json.dumps({k: v for k, v in out.items() if k != "policy"}, indent=1))
    out["rows"] = [{k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
