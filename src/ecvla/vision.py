"""Frozen image encoder for the `full` vision level of EXP-0001 (A1), and its precomputed per-frame features.

ImageNet ResNet-18, global-average-pooled (512-d per camera), on the agent-view and wrist frames at the dataset's
224 px. Frames come straight from the env observation (LIBERO's OpenGL orientation) both in the dataset and online.
The encoder is frozen; only the policy's projection of these features trains.

Usage: python -m ecvla.vision <phaseA_task.h5> <out.npz> [--max-demos N]   (rows align with prepare.py output)
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from ecvla.build_dataset import decode_jpeg
from ecvla.replay import CAMERAS

FEATURE_DIM = 512 * len(CAMERAS)


class FrozenEncoder:
    def __init__(self, device: str = "cpu"):
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 -> (N, 512) float32."""
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255
        return self.net((x - self.mean) / self.std).float().cpu().numpy()

    def encode_obs(self, obs: dict) -> np.ndarray:
        return np.concatenate([self(obs[key][None])[0] for key in CAMERAS.values()])


def precompute(data_path: str, out_path: str, max_demos: int | None = None, batch: int = 256) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"skip: {out_path} exists")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = FrozenEncoder("cuda" if torch.cuda.is_available() else "cpu")
    feats = []
    with h5py.File(data_path) as f:
        for name in sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:max_demos]:
            per_cam = []
            for cam in CAMERAS:
                frames = np.stack([decode_jpeg(x) for x in f[f"{name}/images/{cam}"][()]])
                per_cam.append(np.concatenate([enc(frames[i : i + batch]) for i in range(0, len(frames), batch)]))
            feats.append(np.concatenate(per_cam, axis=1))
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, vision=np.concatenate(feats).astype(np.float16))
    os.replace(tmp, out_path)
    print("done", out_path, sum(len(x) for x in feats), "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--max-demos", type=int)
    a = ap.parse_args()
    precompute(a.data, a.out, a.max_demos)


if __name__ == "__main__":
    main()
