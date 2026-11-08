"""EXP-0005 stage 1: readout R1 of the frozen fine-tuned `pi0_fast` for every LIBERO demo frame (README §3).

R1 = final-layer hidden state at the position that predicts the first FAST code token, concatenated with the mean of
the final-layer states over the valid image + language positions (2 × 2048). LeRobot's `pi0_fast` trains on
[images, "Task: …, State: …;\\n", BOS, "Action: ", codes, "|"] and decodes from [images, prompt, BOS], so the first
code token is predicted at the last token of "Action: "; that fixed text is fed teacher-forced (one forward pass, no
sampling). Observations are built as the mixture builder and the slow-only loop build them (`exp0005_common`):
each demo's recorded sim states rendered in our sim at 224 px, state normalized with LIBERO's mixture quantiles.
Frames are indexed as the HDF5s (demo name, step), which `prepared_v2` keeps one-to-one (build_dataset copies
`states` as `sim_state`).

Sanity output per task: the fraction of frames whose argmax at the R1 position is a FAST code token.

Runs in the VLM env (GPU node):
  PYTHONPATH=src uv run --project vlm python scripts/exp0005_latents.py --ckpt <.../pretrained_model> \
    --mixture <mixture dir> --suite libero_spatial --out-dir <dir> [--task 0] [--max-demos 2] [--dtype bfloat16]
Writes <out-dir>/<suite>/<task file stem>.npz: r1 (frames, 4096) float16, ep_start, demos, meta.
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

# Initialize CUDA before robosuite / MuJoCo (EGL rendering) is imported: CUDA init failed on shared GPU nodes
# otherwise (JOURNAL 2026-11-08).
if torch.cuda.is_available():
    torch.cuda.init()

sys.path.insert(0, str(Path(__file__).parent))
from exp0005_common import IMG, IMG_SIZE, libero_images, libero_state  # noqa: E402
from exp0005_slow_only import load_policy, to_tensor  # noqa: E402
from ecvla.libero_env import make_env

S = Path(os.environ.get("ECVLA_SCRATCH", "data/ecvla_scratch"))


@torch.no_grad()
def r1_batch(policy, pre, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """(R1 (B, 2 · width) float32, argmax token id at the R1 position (B,))."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

    b = pre(batch)
    model = policy.model
    images, img_masks = policy._preprocess_images(b)
    tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
    tok = model._require_paligemma_tokenizer()
    lead = [tok.bos_token_id] + tok.encode("Action: ", add_special_tokens=False)
    fast = torch.tensor(lead, device=tokens.device)[None].expand(tokens.shape[0], -1)
    embs, pad, att, _, n_fast = model.embed_prefix_fast(images, img_masks, tokens, masks, fast_action_tokens=fast,
                                                        fast_action_masks=torch.ones_like(fast, dtype=torch.bool))
    lm = model.paligemma_with_expert.paligemma.model.language_model
    if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        embs = embs.to(torch.bfloat16)
    (out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(att, dtype=embs.dtype), position_ids=torch.cumsum(pad, 1) - 1,
        past_key_values=None, inputs_embeds=[embs, None], use_cache=False, adarms_cond=[None, None])
    last = out[:, -1].float()
    pm = pad[:, :-n_fast].float()
    pooled = (out[:, :-n_fast].float() * pm[..., None]).sum(1) / pm.sum(1, keepdim=True)
    arg = model.paligemma_with_expert.paligemma.lm_head(out[:, -1:]).argmax(-1)[:, 0]
    return torch.cat([last, pooled], 1), arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", type=int, nargs="*", help="task file indices (sorted); default all")
    ap.add_argument("--max-demos", type=int, default=0, help="per task; 0 = all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    a = ap.parse_args()
    from libero.libero import get_libero_path

    mix = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    stats = mix["sources"]["libero"]
    slo, shi = np.array(stats["state_q01"]), np.array(stats["state_q99"])
    fmt = mix.get("prompt_format", "{task}; {condition}")
    policy, pre, _, pinfo = load_policy(a.ckpt, a.dtype)
    print("policy:", pinfo, flush=True)
    files = sorted(glob.glob(str(S / "raw" / a.suite / "*.hdf5")))
    for i in a.task if a.task is not None else range(len(files)):
        f = files[i]
        out_path = Path(a.out_dir) / a.suite / (Path(f).stem.removesuffix("_demo") + ".npz")
        if out_path.exists():
            print("skip", out_path)
            continue
        bddl = os.path.join(get_libero_path("bddl_files"), a.suite, Path(f).stem.removesuffix("_demo") + ".bddl")
        env = make_env(bddl, image_size=IMG_SIZE, cameras=True)  # as the mixture builder renders LIBERO
        t0, rows, args, lengths = time.time(), [], [], []
        with h5py.File(f, "r") as h:
            task = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
            text = fmt.format(condition=stats["condition"], task=task)
            demos = sorted(h["data"], key=lambda s: int(s.split("_")[1]))[: a.max_demos or None]
            for name in demos:
                states = h["data"][name]["states"][:]
                lengths.append(len(states))
                for s0 in range(0, len(states), a.batch):
                    imgs, sts = [], []
                    for s in states[s0: s0 + a.batch]:
                        obs = env.set_init_state(s)
                        imgs.append(libero_images(obs))
                        sts.append(np.clip(2 * (libero_state(obs) - slo) / np.maximum(shi - slo, 1e-6) - 1, -1, 1))
                    batch = {IMG[0]: torch.cat([to_tensor(m) for m, _ in imgs]),
                             IMG[1]: torch.cat([to_tensor(w) for _, w in imgs]),
                             "observation.state": torch.from_numpy(np.stack(sts)).float(), "task": [text] * len(sts)}
                    r1, arg = r1_batch(policy, pre, batch)
                    rows.append(r1.cpu().numpy().astype(np.float16))
                    args.append(arg.cpu().numpy())
        env.close()
        r1, arg = np.concatenate(rows), np.concatenate(args)
        act = policy._paligemma_tokens_to_act_tokens(torch.from_numpy(arg)).numpy()  # FAST+ code ids are 0..2047
        is_fast = float(((act >= 0) & (act < 2048)).mean())
        meta = {"ckpt": a.ckpt, "policy": pinfo, "suite": a.suite, "task": task, "source_file": f, "prompt": text,
                "readout": "R1: [h(last 'Action: ' token), mean h(valid image+language)]", "width": r1.shape[1] // 2,
                "argmax_is_fast_frac": is_fast, "batch": a.batch}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez(tmp, r1=r1, argmax=arg, ep_start=np.r_[0, np.cumsum(lengths)], demos=np.asarray(demos),
                 meta=np.asarray(json.dumps(meta)))
        os.replace(tmp, out_path)
        print(f"done {out_path} frames={len(r1)} demos={len(demos)} {time.time() - t0:.0f}s "
              f"argmax_is_fast={is_fast:.3f} finite={bool(np.isfinite(r1).all())}", flush=True)


if __name__ == "__main__":
    main()
