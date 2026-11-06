"""EXP-0005: does bf16 change what the fine-tuned `pi0_fast` outputs, compared with fp32? Decides the dtype for
slow-only and the R1 latents (both must use the same one).

On the same LIBERO demo frames (rendered from the recorded states as in the mixture), the model is run in fp32,
then cast to bf16 in place (LeRobot's selective cast: vision tower, projector and norms stay fp32) and run again.
Per frame: the decoded FAST chunk (`predict_chunk`, greedy) and R1 (`r1_batch`, batch 1). Reported: fraction of
frames whose decoded actions are identical, action L1 bf16 vs fp32 in normalized units (reference: FAST+
round-trip L1 0.017 normalized, README change log 2026-11-03) and against the demo's recorded actions for both
dtypes, R1 cosine similarity and relative L2 per half (last-token state, pooled prefix).

  PYTHONPATH=src uv run --project vlm python scripts/exp0005_dtype_check.py --ckpt <.../pretrained_model> \
    --mixture <mixture dir> --out <json> [--tasks 0 5] [--stride 10]
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from exp0005_common import IMG, IMG_SIZE, libero_images, libero_state  # noqa: E402
from exp0005_latents import S, r1_batch  # noqa: E402
from exp0005_slow_only import load_policy, predict_chunk, to_tensor  # noqa: E402
from ecvla.libero_env import make_env


def frames(mix, suites, tasks, stride):
    """(suite, task file, demo, step, batch, recorded normalized action chunk) for demo_0 of each task, every stride."""
    from libero.libero import get_libero_path

    st = mix["sources"]["libero"]
    slo, shi = np.array(st["state_q01"]), np.array(st["state_q99"])
    alo, ahi = np.array(st["action_q01"]), np.array(st["action_q99"])
    fmt = mix.get("prompt_format", "{task}; {condition}")
    for suite in suites:
        files = sorted(glob.glob(str(S / "raw" / suite / "*.hdf5")))
        for i in tasks:
            stem = Path(files[i]).stem.removesuffix("_demo")
            env = make_env(os.path.join(get_libero_path("bddl_files"), suite, stem + ".bddl"), image_size=IMG_SIZE,
                           cameras=True)
            with h5py.File(files[i], "r") as h:
                text = fmt.format(condition=st["condition"],
                                  task=json.loads(h["data"].attrs["problem_info"])["language_instruction"])
                d = h["data"]["demo_0"]
                states, acts = d["states"][:], d["actions"][:]
            for t in range(0, len(states), stride):
                obs = env.set_init_state(states[t])
                m, w = libero_images(obs)
                s = np.clip(2 * (libero_state(obs) - slo) / np.maximum(shi - slo, 1e-6) - 1, -1, 1)
                batch = {IMG[0]: to_tensor(m), IMG[1]: to_tensor(w),
                         "observation.state": torch.from_numpy(s).float()[None], "task": [text]}
                rec = np.clip(2 * (acts[t: t + 20] - alo) / np.maximum(ahi - alo, 1e-6) - 1, -1, 1)
                yield suite, stem, t, batch, rec
            env.close()


def run(policy, pre, post, items):
    out = []
    for *_, batch, _ in items:
        r1, _ = r1_batch(policy, pre, batch)
        out.append((predict_chunk(policy, pre, post, batch), r1[0].cpu().numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    mix = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    items = list(frames(mix, ("libero_spatial", "libero_object"), a.tasks, a.stride))
    print(f"{len(items)} frames", flush=True)
    policy, pre, post, info = load_policy(a.ckpt, "float32")
    t0 = time.time()
    f32 = run(policy, pre, post, items)
    t32 = (time.time() - t0) / len(items)
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params(torch.bfloat16)
    t0 = time.time()
    b16 = run(policy, pre, post, items)
    t16 = (time.time() - t0) / len(items)
    rows = []
    for (suite, stem, t, _, rec), (c32, z32), (c16, z16) in zip(items, f32, b16):
        n = min(len(rec), len(c32), len(c16))
        w = len(z32) // 2
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))  # noqa: E731
        rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(u))  # noqa: E731
        l1 = lambda u, v: None if u is None or v is None else float(np.abs(u[:len(v)] - v[:len(u)]).mean())  # noqa: E731
        rows.append({"suite": suite, "task": stem, "step": t, "fail_fp32": c32 is None, "fail_bf16": c16 is None,
                     "identical": c32 is not None and c16 is not None and bool(np.array_equal(c32, c16)),
                     "l1_bf16_vs_fp32": l1(c32, c16), "l1_fp32_vs_demo": l1(c32, rec), "l1_bf16_vs_demo": l1(c16, rec),
                     "r1_last_cos": cos(z32[:w], z16[:w]), "r1_pool_cos": cos(z32[w:], z16[w:]),
                     "r1_last_rel_l2": rel(z32[:w], z16[:w]), "r1_pool_rel_l2": rel(z32[w:], z16[w:])})
    keys = [k for k in rows[0] if k not in ("suite", "task", "step")]
    mean = lambda k: float(np.mean(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None  # noqa: E731
    summary = {k: mean(k) for k in keys}
    summary.update({f"{k}_min": float(np.min([r[k] for r in rows])) for k in ("r1_last_cos", "r1_pool_cos")})
    l1s = [r["l1_bf16_vs_fp32"] for r in rows if r["l1_bf16_vs_fp32"] is not None]
    summary.update({"l1_bf16_vs_fp32_p95": float(np.quantile(l1s, 0.95)) if l1s else None,
                    "s_per_frame_fp32": t32, "s_per_frame_bf16": t16, "frames": len(rows)})
    print(json.dumps(summary, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "policy": info, "summary": summary, "frames": rows}, indent=1))


if __name__ == "__main__":
    main()
