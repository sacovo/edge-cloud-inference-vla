"""EXP-0005: write a LeRobot `pi0_fast` checkpoint whose VLM is `pi05_base`'s, as the fine-tuning start point.

`lerobot_train --policy.path=<out>` then starts from it. Loading as in the stage-0 load check (manual, with a
missing-key check, since `PI0FastPolicy.from_pretrained` only warns on errors); pi05's action expert is dropped.
Features match the mixture (`mixture_stats.json`: state / action dims, two cameras); normalization is identity
because the mixture is normalized per source.

Runs in the VLM env:  uv run --project vlm python scripts/exp0005_make_init_ckpt.py --mixture <dataset dir> --out <dir>
"""

import argparse
import json
from pathlib import Path

CHUNK = 20  # 1 s at 20 Hz (FAST+ card: 1 s chunks)
IMG = ("observation.images.image", "observation.images.wrist_image")
EXPERT = ("paligemma_with_expert.gemma_expert.", "action_in_proj.", "action_out_proj.", "time_mlp_in.",
          "time_mlp_out.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    stats = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    cfg = PI0FastConfig(paligemma_variant="gemma_2b", chunk_size=CHUNK, n_action_steps=CHUNK, device="cpu")
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                 "ACTION": NormalizationMode.IDENTITY}
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(stats["state_dim"],)),
                          **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in IMG}}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(stats["action_dim"],))}
    policy = PI0FastPolicy(cfg)
    sd = policy._fix_pytorch_state_dict_keys(load_file(cached_file("lerobot/pi05_base", "model.safetensors")), cfg)
    sd = {k if k.startswith("model.") else f"model.{k}": v for k, v in sd.items()}
    res = policy.load_state_dict(sd, strict=False)
    bad = [k for k in res.unexpected_keys if not k.removeprefix("model.").startswith(EXPERT)]
    assert not res.missing_keys and not bad, (res.missing_keys[:10], bad[:10])
    policy.save_pretrained(a.out)
    print(f"saved {a.out}: {len(policy.state_dict())} tensors, dropped {len(res.unexpected_keys)} action-expert tensors")


if __name__ == "__main__":
    main()
