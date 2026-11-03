"""EXP-0005 stage 0: does `pi05_base`'s LM head already predict FAST+ tokens in `pi0_fast`'s vocab mapping?

Teacher-forced `pi0_fast` training loss (cross-entropy over FAST action tokens, the full PaliGemma vocab) on LIBERO
frames, no training:
  A  pi05_base weights, true action chunks
  B  pi05_base weights, action chunks shuffled across frames (does the prediction use the observation?)
  C  plain PaliGemma (google/paligemma-3b-pt-224, never saw FAST tokens), true action chunks
References: uniform over the full vocab, ln(257152) = 12.46; uniform over the 2048 FAST tokens, ln(2048) = 7.62.
A << C means the head knows the mapping; A < B means it reads the observation. The loss is `pi0_fast`'s own
(masked next-token CE over the action-token sequence, which may include fixed prefix / end tokens besides the FAST
tokens; they enter A, B and C alike).

Inputs as openpi's LIBERO pipeline (`examples/libero/main.py`): both cameras rotated 180° (the HDF5s are in the
OpenGL convention) and resized to 224; state = eef pos (3) + eef axis-angle (3, HDF5 `ee_ori`) + gripper qpos (2);
actions are 1 s chunks (20 steps at LIBERO's 20 Hz). State and actions are normalized here per dimension with the
1% / 99% quantiles of the sampled suites (the planned per-dataset recipe), so LeRobot's normalizer is identity.

Runs in the VLM env on a GPU node:  uv run --project vlm python scripts/exp0005_fast_ce.py --out <json>
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
SUITES = ("libero_spatial", "libero_object")
CHUNK = 20
IMG = ("observation.images.image", "observation.images.wrist_image")


def load_frames(n, seed):
    """n random (frame, 1 s chunk) samples from the raw demos, plus quantile stats over all frames."""
    rng = np.random.default_rng(seed)
    files = sorted(f for su in SUITES for f in glob.glob(str(S / "raw" / su / "*.hdf5")))
    all_act, all_state, demos = [], [], []
    for f in files:
        h = h5py.File(f, "r")
        task = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
        for name in h["data"]:
            d = h["data"][name]
            state = np.concatenate([d["obs/ee_pos"][:], d["obs/ee_ori"][:], d["obs/gripper_states"][:]], 1)
            all_act.append(d["actions"][:])
            all_state.append(state)
            demos.append((f, name, task, len(state)))
        h.close()
    act, st = np.concatenate(all_act), np.concatenate(all_state)
    q = {"action": (np.quantile(act, 0.01, 0), np.quantile(act, 0.99, 0)),
         "state": (np.quantile(st, 0.01, 0), np.quantile(st, 0.99, 0))}
    norm = lambda x, k: np.clip(2 * (x - q[k][0]) / np.maximum(q[k][1] - q[k][0], 1e-6) - 1, -1, 1)  # noqa: E731
    out = {IMG[0]: [], IMG[1]: [], "observation.state": [], "action": [], "task": []}
    for i in rng.choice(len(demos), n):
        f, name, task, T = demos[i]
        t = int(rng.integers(0, T - CHUNK))
        with h5py.File(f, "r") as h:
            d = h["data"][name]
            for key, cam in zip(IMG, ("agentview_rgb", "eye_in_hand_rgb")):
                img = np.ascontiguousarray(d["obs"][cam][t][::-1, ::-1])  # 180°, as openpi
                x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255
                out[key].append(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)[0])
            state = np.concatenate([d["obs/ee_pos"][t], d["obs/ee_ori"][t], d["obs/gripper_states"][t]])
            out["observation.state"].append(torch.from_numpy(norm(state, "state")).float())
            out["action"].append(torch.from_numpy(norm(d["actions"][t: t + CHUNK], "action")).float())
            out["task"].append(task)
    return {k: (torch.stack(v) if k != "task" else v) for k, v in out.items()}, len(demos)


def build_policy(device):
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from lerobot.policies.pi0_fast.processor_pi0_fast import make_pi0_fast_pre_post_processors

    cfg = PI0FastConfig(paligemma_variant="gemma_2b", chunk_size=CHUNK, n_action_steps=CHUNK, device=device)
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                 "ACTION": NormalizationMode.IDENTITY}
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                          **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in IMG}}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = PI0FastPolicy(cfg).eval()
    pre, _ = make_pi0_fast_pre_post_processors(cfg, dataset_stats=None)
    return policy, pre


def load_weights(policy, which):
    """'pi05': lerobot/pi05_base (as the load check); 'paligemma': HF google/paligemma-3b-pt-224."""
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    if which == "pi05":
        sd = policy._fix_pytorch_state_dict_keys(load_file(cached_file("lerobot/pi05_base", "model.safetensors")),
                                                 policy.config)
        sd = {k if k.startswith("model.") else f"model.{k}": v for k, v in sd.items()}
    else:
        from transformers import PaliGemmaForConditionalGeneration

        hf = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224", torch_dtype=torch.float32)
        V = policy.model.paligemma_with_expert.paligemma.lm_head.weight.shape[0]  # 257152; HF pads to 257216
        sd = {f"model.paligemma_with_expert.paligemma.{k}": (v[:V] if v.shape[0] > V and v.ndim == 2 and
                                                              ("lm_head" in k or "embed_tokens" in k) else v)
              for k, v in hf.state_dict().items()}
        del hf
    res = policy.load_state_dict(sd, strict=False)
    own = set(policy.state_dict())
    loaded = len(own & set(sd))
    return {"loaded": loaded, "model_tensors": len(own), "missing": res.missing_keys[:20],
            "n_missing": len(res.missing_keys), "n_unexpected": len(res.unexpected_keys)}


@torch.no_grad()
def score(policy, batch, bs):
    """Mean next-token CE (nats) over the action-token sequence, weighted as `pi0_fast` masks it (targets 1:)."""
    from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, \
        OBS_LANGUAGE_TOKENS

    m = policy.model
    tot_ce, tot_n, losses = 0.0, 0, []
    for i in range(0, len(batch[ACTION_TOKENS]), bs):
        b = {k: (v[i: i + bs] if torch.is_tensor(v) else v) for k, v in batch.items()}
        images, img_masks = policy._preprocess_images(b)
        out = m.forward(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK],
                        b[ACTION_TOKENS], b[ACTION_TOKEN_MASK])
        losses.append(float(out["ce_loss"]))
        n = int(b[ACTION_TOKEN_MASK][:, 1:].sum())
        tot_ce += float(out["ce_loss"]) * n
        tot_n += n
    return {"ce_mean_token": tot_ce / tot_n, "tokens": tot_n, "ce_per_batch": losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, n_demos = load_frames(a.n, seed=0)
    policy, pre = build_policy(device)
    batch = pre(dict(raw))
    perm = torch.randperm(a.n, generator=torch.Generator().manual_seed(1))
    shuffled = pre({**raw, "action": raw["action"][perm]})
    from lerobot.utils.constants import ACTION_TOKEN_MASK

    out = {"n_frames": a.n, "n_demos": n_demos, "chunk": CHUNK, "device": device,
           "fast_tokens_per_chunk": float(batch[ACTION_TOKEN_MASK].float().sum(1).mean()),
           "uniform_full_vocab": math.log(257152), "uniform_fast_vocab": math.log(2048)}
    out["weights_pi05"] = load_weights(policy, "pi05")
    out["A_pi05_true"] = score(policy, batch, a.bs)
    out["B_pi05_shuffled"] = score(policy, shuffled, a.bs)
    print("A", out["A_pi05_true"]["ce_mean_token"], "B", out["B_pi05_shuffled"]["ce_mean_token"], flush=True)
    try:
        out["weights_paligemma"] = load_weights(policy, "paligemma")
        out["C_paligemma_true"] = score(policy, batch, a.bs)
        print("C", out["C_paligemma_true"]["ce_mean_token"], flush=True)
    except Exception as e:
        out["C_paligemma_true"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
        print("C", out["C_paligemma_true"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    print({k: {kk: vv for kk, vv in v.items() if kk != "ce_per_batch"} for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
