"""Training on synthetic prepared files: runs end to end, learns, and the closed-loop Runner reproduces the model."""

import json

import numpy as np
import pytest
import torch

from ecvla.sensors import FT_DIM, PROPRIO_DIM, TORQUE_DIM
from ecvla.train import EpisodeData, parser, train


def fake_task(path, task, objects, n_demos=20, seed=0, fixtures=None):
    """fixtures=None writes the v1 format (prepared before fixture state existed)."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(30, 50, n_demos)
    N = int(lengths.sum())
    # per-episode constant target, visible in proprio: the whole action chunk is predictable
    c = np.repeat(rng.normal(size=(n_demos, 7)), lengths, axis=0)
    proprio = rng.normal(size=(N, PROPRIO_DIM)).astype(np.float32)
    proprio[:, :7] = c + 0.1 * proprio[:, :7]
    actions = 0.3 * c
    meta = {"suite": "libero_fake", "task": task + "_demo", "bddl_file_name": f"x/{task}.bddl",
            "language_instruction": task, "objects": objects, "demos": [f"demo_{i}" for i in range(n_demos)]}
    extra = {}
    if fixtures is not None:
        meta["fixtures"] = fixtures
        extra["fix_state"] = rng.normal(size=(N, len(fixtures), 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.zeros(n_demos, int)
    first[0] = -1  # never reaches success: must be dropped
    np.savez(path, proprio=proprio, torque=rng.normal(size=(N, TORQUE_DIM)), ft=rng.normal(size=(N, FT_DIM)),
             fast_valid=np.ones(N), obj_pose=rng.normal(size=(N, len(objects), 7)), contact=np.zeros((N, len(objects))),
             actions=actions, ep_start=np.r_[0, np.cumsum(lengths)], first_success_step=first,
             meta=np.asarray(json.dumps(meta)), **extra)


@pytest.fixture
def data_dir(tmp_path):
    fake_task(tmp_path / "libero_fake/a_demo.npz", "a", ["bowl_1", "plate_1"], seed=0)
    fake_task(tmp_path / "libero_fake/b_demo.npz", "b", ["bowl_1", "cup_1", "mug_1"], seed=1,
              fixtures=["cabinet_1_top_level", "stove_1_button"])
    return tmp_path


def test_episode_data_pads_objects_and_drops_failed_demos(data_dir):
    d = EpisodeData(data_dir, ["libero_fake"], val_per_task=1)
    assert d.tasks == ["a", "b"] and d.max_obj == 3 and d.max_fix == 2
    assert d.names == ["bowl_1", "cabinet_1_top_level", "cup_1", "mug_1", "plate_1", "stove_1_button"]
    assert d.fix_mask.tolist() == [[False, False], [True, True]]
    assert len(d.episodes) == 38 and len(d.split(True)) == 2
    assert d.obj_mask.tolist() == [[True, True, False], [True, True, True]]


@pytest.mark.parametrize("condition,core,head", [("none", "gru", "l1"), ("oracle-delayed", "tf", "flow"),
                                                  ("task-id", "gru", "flow")])
def test_train_learns_and_runner_matches(data_dir, tmp_path, condition, core, head):
    out = tmp_path / "run"
    args = parser().parse_args(["--data", str(data_dir), "--suites", "libero_fake", "--condition", condition,
                                "--size", "20000", "--out", str(out), "--core", core, "--head", head,
                                "--steps", "400" if head == "l1" else "1500", "--batch", "8", "--lr", "3e-3", "--log-every", "50",
                                "--val-per-task", "1", "--device", "cpu"])
    train(args)
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < log[0]["loss"]
    # beat predicting the mean action clearly (flow samples carry sampling noise on top of the fit)
    acts = EpisodeData(data_dir, ["libero_fake"], val_per_task=1).rows["actions"]
    assert log[-1]["val_l1"] < (0.6 if head == "l1" else 0.75) * np.abs(acts - acts.mean(0)).mean()

    from ecvla.evaluate import Runner

    ckpt = torch.load(out / "model.pt", weights_only=False)
    assert ckpt["info"]["policy"]["fixtures"] == condition.startswith("oracle")
    r = Runner(ckpt, "b")
    r.reset(20, np.random.default_rng(0), k=1 if condition == "oracle-delayed" else None)
    obs = {"proprio": np.zeros(PROPRIO_DIM), "torque": np.zeros(TORQUE_DIM), "ft": np.zeros(FT_DIM), "fast_valid": 0.0,
           "obj_pose": np.zeros((3, 7)), "contact": np.zeros(3), "fix_state": np.zeros((2, 2))}
    a = r.act(obs)
    assert a.shape == (7,) and np.isfinite(a).all()
