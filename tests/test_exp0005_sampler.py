"""EXP-0005 weighted sampler (scripts/exp0005_train.py), against a stand-in for LeRobot's EpisodeAwareSampler."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("exp0005_train", Path(__file__).parents[1] / "scripts/exp0005_train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


class FakeEpisodeAwareSampler:
    """The parts of LeRobot's EpisodeAwareSampler the subclass relies on (f5c9bc0, datasets/sampler.py)."""

    def __init__(self, from_idx, to_idx, shuffle=True, seed=0):
        self._starts = np.asarray(from_idx)
        self._cum_lengths = np.cumsum(np.asarray(to_idx) - np.asarray(from_idx))
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle, self.seed = shuffle, seed

    def _epoch_generator(self, epoch):
        return torch.Generator().manual_seed(int(np.random.SeedSequence([self.seed, epoch]).generate_state(1)[0]))

    def _frame_index(self, position):
        e = int(np.searchsorted(self._cum_lengths, position, side="right"))
        return int(self._starts[e]) + position - (int(self._cum_lengths[e - 1]) if e > 0 else 0)

    def _iter_epoch(self, epoch, start):
        yield from (self._frame_index(k) for k in range(start, self._num_frames))


def test_weighted_sampler_realizes_the_source_shares_and_resumes_deterministically():
    # three sources: libero (2 eps x 100 frames), a (1 x 900), b (3 x 1000) -> frames 200 / 900 / 3000
    lengths = [100, 100, 900, 1000, 1000, 1000]
    to_idx = np.cumsum(lengths)
    from_idx = to_idx - lengths
    T._WEIGHTS = {"sources": ["libero", "a", "b"], "episodes": [2, 1, 3], "libero_share": 0.3}
    cls = T.make_sampler_class(FakeEpisodeAwareSampler)
    s = cls(from_idx, to_idx)
    want = T.source_shares({"libero": 200, "a": 900, "b": 3000}, 0.3)
    assert abs(s.shares["libero"] - 0.3) < 1e-12 and abs(sum(s.shares.values()) - 1) < 1e-12
    assert abs(want["a"] / want["b"] - (900 / 3000) ** T.POWER) < 1e-9
    idx = np.array(list(s._iter_epoch(0, 0)))
    assert len(idx) == 4100 and idx.min() >= 0 and idx.max() < 4100
    src = np.digitize(idx, [200, 1100])  # frame index -> source
    for k, name in enumerate(["libero", "a", "b"]):
        assert abs((src == k).mean() - want[name]) < 0.03, (name, (src == k).mean(), want[name])
    assert (idx[src == 0] < 200).all() and len(np.unique(idx[src == 0])) > 100  # libero frames spread over both eps
    np.testing.assert_array_equal(np.array(list(s._iter_epoch(0, 100))), idx[100:])  # resume from an offset
    assert not np.array_equal(np.array(list(s._iter_epoch(1, 0))), idx)  # next epoch differs
