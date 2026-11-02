"""EXP-0005 stage 0: can LeRobot's `pi0_fast` (FAST-token training) start from `pi05_base`'s VLM weights?

Checks, in order (each prints PASS / FAIL and the evidence):
  1. the gated PaliGemma tokenizer is reachable (`pi0_fast` needs it at init);
  2. `pi05_base` loads into `pi0_fast` with no missing keys; the unused tensors are only pi05's action expert;
  3. FAST+ (`lerobot/fast-action-tokenizer`, byte-identical to `physical-intelligence/fast`) round-trips LIBERO
     action chunks (1 s = 20 steps, per-dimension 1% / 99% quantiles to [-1, 1]): tokens per chunk and
     reconstruction error, and the PaliGemma ids they map to.
The weights are loaded here, not via `from_pretrained`, because that method catches load errors and only prints a
warning (`modeling_pi0_fast.py`, end of `from_pretrained`).

Runs in the VLM env:  uv run --project vlm python scripts/exp0005_load_check.py --out <json>
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
SUITES = ("libero_spatial", "libero_object")
CHUNK = 20  # 1 s at LIBERO's 20 Hz (FAST+ card recommends 1 s chunks)
EXPERT = ("paligemma_with_expert.gemma_expert.", "action_in_proj.", "action_out_proj.", "time_mlp_in.",
          "time_mlp_out.")


def check_tokenizer_access(out):
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        out["paligemma_tokenizer"] = "PASS"
    except Exception as e:  # gated repo: needs an HF token with the license accepted
        out["paligemma_tokenizer"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
    print("1 paligemma tokenizer:", out["paligemma_tokenizer"], flush=True)
    return out["paligemma_tokenizer"] == "PASS"


def check_load(out):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    cfg = PI0FastConfig(paligemma_variant="gemma_2b", chunk_size=CHUNK, n_action_steps=CHUNK, device="cpu")
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                          "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = PI0FastPolicy(cfg)
    path = cached_file("lerobot/pi05_base", "model.safetensors")
    sd = load_file(path)
    fixed = policy._fix_pytorch_state_dict_keys(sd, cfg)
    remapped = {k if k.startswith("model.") else f"model.{k}": v for k, v in fixed.items()}
    res = policy.load_state_dict(remapped, strict=False)
    expected = set(policy.state_dict())
    unexpected_non_expert = [k for k in res.unexpected_keys if not k.removeprefix("model.").startswith(EXPERT)]
    shape_ok = all(remapped[k].shape == policy.state_dict()[k].shape for k in remapped if k in expected)
    out["load"] = {
        "checkpoint": path, "checkpoint_tensors": len(sd), "model_tensors": len(expected),
        "missing": res.missing_keys, "unexpected": len(res.unexpected_keys),
        "unexpected_non_expert": unexpected_non_expert, "shapes_match": shape_ok,
        "verdict": "PASS" if not res.missing_keys and not unexpected_non_expert and shape_ok else "FAIL",
    }
    print("2 load:", out["load"]["verdict"], {k: v for k, v in out["load"].items() if k != "missing"},
          "missing:", res.missing_keys[:10], flush=True)
    return policy


def check_fast(out, policy):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)
    acts = np.concatenate([np.load(f)["actions"] for su in SUITES
                           for f in sorted(glob.glob(str(S / "a1/prepared_v2" / su / "*.npz")))]).astype(np.float32)
    lo, hi = np.quantile(acts, 0.01, axis=0), np.quantile(acts, 0.99, axis=0)
    norm = np.clip(2 * (acts - lo) / np.maximum(hi - lo, 1e-6) - 1, -1, 1)
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(norm) - CHUNK, 512)  # chunks may cross demo boundaries; fine for a codec check
    chunks = np.stack([norm[s: s + CHUNK] for s in starts])
    tokens = proc(chunks)
    dec = np.asarray(proc.decode(tokens, time_horizon=CHUNK, action_dim=7))
    lens = np.array([len(t) for t in tokens])
    err = np.abs(dec - chunks)
    vocab = policy._paligemma_tokenizer.vocab_size
    flat = np.concatenate([np.asarray(t) for t in tokens])
    pg = vocab - 1 - policy.config.fast_skip_tokens - flat  # inverse of `_paligemma_tokens_to_act_tokens`
    out["fast"] = {
        "chunks": len(chunks), "tokens_per_chunk_mean": float(lens.mean()), "tokens_per_chunk_max": int(lens.max()),
        "max_action_tokens": policy.config.max_action_tokens,
        "recon_l1_norm_mean": float(err.mean()), "recon_l1_norm_max": float(err.max()),
        "recon_l1_raw_mean": float((err * (hi - lo) / 2).mean()),
        "action_token_range": [int(flat.min()), int(flat.max())], "paligemma_id_range": [int(pg.min()), int(pg.max())],
        "paligemma_vocab": vocab, "fast_skip_tokens": policy.config.fast_skip_tokens,
        "quantiles_lo": lo.tolist(), "quantiles_hi": hi.tolist(),
    }
    print("3 fast:", {k: v for k, v in out["fast"].items() if not k.startswith("quantiles")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    out = {}
    try:
        if check_tokenizer_access(out):
            check_fast(out, check_load(out))
    finally:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
