"""Contract of the simulated latent channel: what the fast policy may see at step t."""

import numpy as np
import pytest

from ecvla.latency import DelayModel, episode_start, schedule, seconds_to_steps


def rng():
    return np.random.default_rng(0)


def test_fixed_delay_hand_computed():
    # queries at 0, 4, 8; each arrives 2 steps later
    s = schedule(10, period=4, delay=DelayModel(base=2), rng=rng())
    assert s.src.tolist() == [-1, -1, 0, 0, 0, 0, 4, 4, 4, 4]
    assert s.age.tolist() == [-1, -1, 2, 3, 4, 5, 2, 3, 4, 5]


@pytest.mark.parametrize("period,base,jitter", [(1, 0, 0), (1, 3, 5), (4, 2, 10), (10, 1, 0)])
def test_never_sees_unarrived_or_backwards(period, base, jitter):
    s = schedule(300, period, DelayModel(base=base, jitter=jitter), rng())
    t = np.arange(300)
    seen = s.src >= 0
    assert (s.src[seen] <= t[seen] - base).all()  # no frame newer than the minimum delay allows
    assert (np.diff(s.src) >= 0).all()  # z never goes back in time, even when jitter reorders arrivals
    assert (s.age[seen] >= base).all() and (s.age[~seen] == -1).all()


def test_all_dropped_means_no_latent():
    s = schedule(50, period=2, delay=DelayModel(base=1, p_drop=1.0), rng=rng())
    assert (s.src == -1).all()


def test_episode_start_holds_first_frame():
    s = episode_start(20, DelayModel(base=3), rng())
    assert s.src.tolist() == [-1] * 3 + [0] * 17


def test_seconds_to_steps_rounds_up_to_next_control_step():
    assert seconds_to_steps([0.0, 0.05, 0.051, 0.2, 0.249], 20).tolist() == [0, 1, 2, 4, 5]
