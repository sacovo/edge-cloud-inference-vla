"""Fast-policy contract: head-size targets, causality, stepwise = sequence (closed loop matches training), masking."""

import numpy as np
import pytest
import torch

from ecvla.policy import FastPolicy, PolicyConfig, WINDOW, config_for_size, count_head_params
from ecvla.sensors import FT_DIM, PROPRIO_DIM, TORQUE_DIM
from ecvla.train import state_source

ARCHS = [("gru", "l1"), ("tf", "l1"), ("gru", "flow"), ("tf", "flow")]


def batch(B=2, T=40, S=5, vision=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    b = {"proprio": r(B, T, PROPRIO_DIM), "torque": r(B, T, TORQUE_DIM), "ft": r(B, T, FT_DIM),
         "fast_valid": torch.ones(B, T), "task": torch.arange(B) % 3, "obj_name": torch.arange(S).repeat(B, 1),
         "obj_mask": torch.ones(B, S, dtype=torch.bool), "obj_pose": r(B, T, S, 7), "contact": torch.zeros(B, T, S),
         "state_avail": torch.ones(B, T, dtype=torch.bool), "age": torch.zeros(B, T),
         "fix_state": r(B, T, 2, 2), "fix_name": torch.tensor([[5, 6]]).repeat(B, 1), "fix_mask": torch.ones(B, 2, dtype=torch.bool)}
    if vision:
        b["vision"] = r(B, T, vision)
    return b


def model(core, head, cond="state", d=32, layers=2):
    torch.manual_seed(0)
    cfg = PolicyConfig(d=d, layers=layers, cond=cond, n_tasks=3, n_names=7, core=core, head=head, fixtures=cond == "state")
    return FastPolicy(cfg).eval()


@pytest.mark.parametrize("core,head", ARCHS)
@pytest.mark.parametrize("cond", ["none", "task", "state"])
def test_head_size_hits_target(core, head, cond):
    for target in (10_000, 100_000, 1_000_000, 10_000_000):
        n = count_head_params(config_for_size(target, cond=cond, n_tasks=10, n_names=40, core=core, head=head))
        assert 0.7 < n / target < 1.4, (target, n)


@pytest.mark.parametrize("core,head", ARCHS)
def test_future_inputs_do_not_change_past_features(core, head):
    m, b = model(core, head), batch()
    b2 = {k: v.clone() for k, v in b.items()}
    for k in ("proprio", "torque", "ft", "obj_pose", "fix_state"):
        b2[k][:, 25:] += 5.0
    with torch.no_grad():
        f1, f2 = m.features(b)[0], m.features(b2)[0]
    torch.testing.assert_close(f1[:, :25], f2[:, :25])
    assert not torch.allclose(f1[:, 25:], f2[:, 25:])


@pytest.mark.parametrize("core,head", ARCHS)
def test_stepwise_matches_sequence(core, head):
    """Closed loop feeds one step at a time; it must give what training computed on the whole episode."""
    m, b = model(core, head), batch(T=WINDOW * 2 + 3)
    per_ep = ("task", "obj_name", "obj_mask", "fix_name", "fix_mask")
    with torch.no_grad():
        full = m.features(b)[0]
        state, steps = None, []
        for t in range(full.shape[1]):
            bt = {k: (v if k in per_ep else v[:, t : t + 1]) for k, v in b.items()}
            f, state = m.features(bt, state)
            steps.append(f)
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-4)


def test_unavailable_state_is_ignored():
    m, b = model("gru", "l1"), batch()
    b["state_avail"][:, :10] = False
    b2 = {k: v.clone() for k, v in b.items()}
    b2["obj_pose"][:, :10] += 3.0
    b2["fix_state"][:, :10] += 3.0
    with torch.no_grad():
        torch.testing.assert_close(m.features(b)[0][:, :10], m.features(b2)[0][:, :10])


def test_fixture_state_reaches_the_policy():
    m, b = model("gru", "l1"), batch()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["fix_state"][..., 0] += 1.0
    with torch.no_grad():
        assert not torch.allclose(m.features(b)[0], m.features(b2)[0])


def test_flow_sampling_is_seeded():
    m, b = model("gru", "flow"), batch()
    with torch.no_grad():
        f = m.features(b)[0]
        a1 = m.act(f, torch.Generator().manual_seed(1))
        a2 = m.act(f, torch.Generator().manual_seed(1))
    torch.testing.assert_close(a1, a2)


def test_state_source_delays():
    rng = np.random.default_rng(0)
    src, age = state_source(50, "oracle-fresh", rng)
    np.testing.assert_array_equal(src, np.arange(50))
    assert (age == 0).all()
    src, _ = state_source(50, "oracle-delayed", rng, k=10, train=False)
    assert (src[:10] == -1).all() and src[10] == 0 and src[19] == 0 and src[20] == 10  # exactly k late
    for _ in range(20):  # training: k-1..k+1 late, never newer than allowed
        src, _ = state_source(60, "oracle-delayed", rng, k=10)
        t = np.arange(60)
        ok = src >= 0
        assert (t[ok] - src[ok] >= 9).all() and (src[:9] == -1).all()
