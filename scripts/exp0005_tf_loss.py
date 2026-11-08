"""EXP-0005 diagnosis: does the fine-tuned `pi0_fast` use the observation? Teacher-forced training loss on demo frames.

The checkpoint's own pre-processor turns (images, state, prompt, demo action chunk) into the training batch (FAST
tokens included) and `policy.forward` returns the training loss (nats per action token), for:
  true      the frame's own demo chunk (what training saw)
  shuffled  the chunk of a frame from another task (the observation no longer matches the target)
  blank     the true chunk with both images set to black (does the prediction use the images?)
true << shuffled: the model reads the observation, and a decode-side problem would explain the poor decoded chunks;
true ≈ shuffled: it predicts an observation-independent marginal, i.e. the training did not make it condition.
Frames as `exp0005_offline_check` (demo_0, every --stride steps, rendered as in the mixture); only full 20-step chunks.

  PYTHONPATH=src uv run --project vlm python scripts/exp0005_tf_loss.py --ckpt <.../pretrained_model> \
    --mixture <mixture dir> --out <json> [--tasks 0 2 4 6 8] [--stride 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Initialize CUDA before robosuite / MuJoCo (EGL rendering) is imported (JOURNAL 2026-11-08).
if torch.cuda.is_available():
    torch.cuda.init()

sys.path.insert(0, str(Path(__file__).parent))
from exp0005_common import IMG  # noqa: E402
from exp0005_dtype_check import frames  # noqa: E402
from exp0005_slow_only import load_policy  # noqa: E402


@torch.no_grad()
def loss(policy, pre, batch, chunk, ad):
    a = np.zeros((1, len(chunk), ad), np.float32)
    a[0, :, : chunk.shape[1]] = chunk
    b = pre({**batch, "action": torch.from_numpy(a)})
    return float(policy.forward(b)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    mix = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    items = [it for it in frames(mix, ("libero_spatial", "libero_object"), a.tasks, a.stride) if len(it[4]) == 20]
    policy, pre, _, info = load_policy(a.ckpt, "float32")
    policy.eval()
    ad = policy.config.output_features["action"].shape[0]
    n = len(items)
    rows = []
    for i, (suite, stem, t, batch, rec) in enumerate(items):
        j = (i + n // 2) % n  # a frame from another task (frames are grouped by task)
        black = {**batch, IMG[0]: torch.zeros_like(batch[IMG[0]]), IMG[1]: torch.zeros_like(batch[IMG[1]])}
        rows.append({"suite": suite, "task": stem, "step": t, "other_task": items[j][1],
                     "true": loss(policy, pre, batch, rec, ad), "shuffled": loss(policy, pre, batch, items[j][4], ad),
                     "blank": loss(policy, pre, black, rec, ad)})
    out = {"ckpt": a.ckpt, "policy": info, "frames": n,
           "mean": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("true", "shuffled", "blank")},
           "frac_true_below_shuffled": round(float(np.mean([r["true"] < r["shuffled"] for r in rows])), 4)}
    print(json.dumps({k: v for k, v in out.items() if k != "policy"}, indent=1))
    out["rows"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
