"""Simulated delivery of the slow model's latent z to the fast policy (EXP-0001 axis D and deployment profiles).

The slow model is queried every `period` control steps with the observation of that step. Each query's result
arrives `delay` steps later (sampled per query), or never (dropped). At every control step the fast policy sees
the newest-observation latent that has arrived so far, plus its age. A result that arrives late never replaces a
latent computed from a newer observation, so jitter can reorder arrivals but never makes z go backwards in time.

The same schedule serves training (index into precomputed per-frame z) and closed-loop evaluation.
All times are in control steps (20 Hz in LIBERO); use `seconds_to_steps` for measured latencies.
"""

from dataclasses import dataclass

import numpy as np


def seconds_to_steps(seconds, control_freq: float = 20.0) -> np.ndarray:
    """A latency of s seconds is usable at the first control step at or after it."""
    return np.ceil(np.asarray(seconds, dtype=float) * control_freq - 1e-9).astype(int)


@dataclass(frozen=True)
class DelayModel:
    """Per-query delay in control steps: base + Uniform{0..jitter}, or sampled from `trace`; dropped with p_drop."""

    base: int = 0
    jitter: int = 0
    p_drop: float = 0.0
    trace: tuple[int, ...] | None = None  # e.g. measured delays (already in steps); overrides base/jitter

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Delays for n queries; -1 marks a dropped query."""
        if self.trace is not None:
            d = rng.choice(np.asarray(self.trace, dtype=int), size=n)
        else:
            d = self.base + rng.integers(0, self.jitter + 1, size=n)
        d = d.astype(int)
        d[rng.random(n) < self.p_drop] = -1
        return d


@dataclass(frozen=True)
class LatentSchedule:
    src: np.ndarray  # (T,) frame index whose latent the policy sees at step t; -1 = nothing arrived yet
    age: np.ndarray  # (T,) t - src[t] in steps; -1 where src == -1


def schedule(T: int, period: int, delay: DelayModel, rng: np.random.Generator, first_query: int = 0) -> LatentSchedule:
    """Which observation's latent is available at each of T control steps."""
    if period < 1:
        raise ValueError("period must be >= 1")
    queries = np.arange(first_query, T, period)
    delays = delay.sample(len(queries), rng)
    arrival = np.where(delays >= 0, queries + delays, T)  # dropped -> never within the episode
    src = np.full(T, -1, dtype=int)
    for q, a in zip(queries, arrival):
        if a < T:
            src[a:] = np.maximum(src[a:], q)  # newest observation wins; late arrivals of old frames are ignored
    age = np.where(src >= 0, np.arange(T) - src, -1)
    return LatentSchedule(src=src, age=age)


def episode_start(T: int, delay: DelayModel, rng: np.random.Generator) -> LatentSchedule:
    """`z-ep0` condition: only the first frame is ever queried; its latent is held once it arrives."""
    return schedule(T, period=T, delay=delay, rng=rng)
