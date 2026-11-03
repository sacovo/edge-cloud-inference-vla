"""EXP-0005: LeRobot `lerobot_train` with per-source weighted sampling over the merged mixture.

LeRobot samples frames uniformly (`EpisodeAwareSampler`; its `sample_weighting` only weights the loss). This wrapper
replaces the sampler in `lerobot.scripts.lerobot_train` with a subclass that, each epoch, draws a source by its weight
and then a frame uniformly within that source, seeded from (seed, epoch) as the original, so resume stays
deterministic. Sources occupy contiguous episode ranges in the merged dataset, in `mixture_stats.json` order with
`episodes_written` episodes each (`exp0005_merge_mixture.py`).

Weights (README §3): LIBERO gets `--libero-share` of the samples; the other sources share the rest in proportion to
frames^0.43 (π0's mixture rule, 2410.24164 §V-A).

Usage (VLM env): python scripts/exp0005_train.py --mixture <dir> [--libero-share 0.33] <lerobot_train args ...>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

POWER = 0.43
_WEIGHTS: dict | None = None  # set in main(): {"sources": [...], "episodes": [...], "libero_share": float}


def source_shares(frames: dict[str, int], libero_share: float) -> dict[str, float]:
    rest = {k: v**POWER for k, v in frames.items() if k != "libero"}
    tot = sum(rest.values())
    shares = {k: (1 - libero_share) * v / tot for k, v in rest.items()}
    if "libero" in frames:
        shares["libero"] = libero_share
    return shares


def make_sampler_class(base):
    class WeightedEpisodeSampler(base):
        """`EpisodeAwareSampler` with per-source weights (see module docstring); same length and resume state."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = _WEIGHTS
            n_eps = len(self._starts)
            assert sum(cfg["episodes"]) == n_eps, (sum(cfg["episodes"]), n_eps, "episode counts do not match")
            ends = np.cumsum(cfg["episodes"])  # episode index ranges per source
            pos_end = self._cum_lengths[ends - 1]
            pos_start = np.concatenate([[0], pos_end[:-1]])
            self._src_start = torch.as_tensor(pos_start)
            self._src_len = torch.as_tensor(pos_end - pos_start)
            shares = source_shares(dict(zip(cfg["sources"], self._src_len.tolist())), cfg["libero_share"])
            self.shares = shares  # from the frames actually in the dataset
            self._src_p = torch.as_tensor([shares[s] for s in cfg["sources"]], dtype=torch.float64)
            logging.info("weighted sampler (share, frames): %s",
                         {s: (round(shares[s], 4), int(n)) for s, n in zip(cfg["sources"], self._src_len.tolist())})

        def _iter_epoch(self, epoch, start):
            if not self.shuffle:
                yield from super()._iter_epoch(epoch, start)
                return
            g = self._epoch_generator(epoch)
            src = torch.multinomial(self._src_p, self._num_frames, replacement=True, generator=g)
            u = torch.rand(self._num_frames, generator=g, dtype=torch.float64)
            pos = self._src_start[src] + (u * self._src_len[src]).long()
            for k in range(start, self._num_frames):
                yield self._frame_index(int(pos[k]))

    return WeightedEpisodeSampler


def main():
    global _WEIGHTS
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--libero-share", type=float, required=True)
    a, rest = ap.parse_known_args()
    stats = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())["sources"]
    names = list(stats)
    _WEIGHTS = {"sources": names, "episodes": [stats[n]["episodes_written"] for n in names],
                "libero_share": a.libero_share}

    import lerobot.scripts.lerobot_train as T

    T.EpisodeAwareSampler = make_sampler_class(T.EpisodeAwareSampler)
    sys.argv = [sys.argv[0], *rest]
    T.main()


if __name__ == "__main__":
    main()
