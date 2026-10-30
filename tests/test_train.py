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


def test_teacher_latents_train_a_student_and_drive_the_runner(data_dir, tmp_path):
    """A1-teacher (H4): teacher z on demo frames aligns with the rows; the student trains on it; in closed loop the
    student reads the teacher's z through the delay schedule."""
    from ecvla.evaluate import Runner
    from ecvla.teacher import Teacher, precompute

    common = ["--data", str(data_dir), "--suites", "libero_fake", "--steps", "60", "--batch", "4", "--log-every", "30",
              "--val-per-task", "1", "--device", "cpu"]
    teacher_dir = tmp_path / "teacher"
    train(parser().parse_args(common + ["--condition", "task-id", "--size", "20000", "--out", str(teacher_dir)]))
    precompute(str(teacher_dir), str(data_dir / "libero_fake"), str(tmp_path / "z/libero_fake"))
    z = np.load(tmp_path / "z/libero_fake/b_demo.npz")
    rows = np.load(data_dir / "libero_fake/b_demo.npz")["ep_start"][-1]
    assert z["z_chunk"].shape == (rows, 56) and z["z_hidden"].shape[0] == rows

    student = tmp_path / "student"
    train(parser().parse_args(common + ["--condition", "z-teacher", "--size", "10000", "--out", str(student),
                                        "--latent-dir", str(tmp_path / "z"), "--teacher", str(teacher_dir)]))
    ckpt = torch.load(student / "model.pt", weights_only=False)
    assert ckpt["info"]["policy"]["cond"] == "latent" and ckpt["info"]["policy"]["latent_dim"] == 56

    r = Runner(ckpt, "b")
    r.reset(20, np.random.default_rng(0), k=4, teacher=Teacher(teacher_dir))
    obs = {"proprio": np.zeros(PROPRIO_DIM), "torque": np.zeros(TORQUE_DIM), "ft": np.zeros(FT_DIM), "fast_valid": 0.0,
           "obj_pose": np.zeros((3, 7)), "contact": np.zeros(3), "fix_state": np.zeros((2, 2))}
    for _ in range(6):
        a = r.act(obs)
    assert np.isfinite(a).all() and len(r.online_teacher.history) == 6
    np.testing.assert_array_equal(r.src[:6], [-1, -1, -1, -1, 0, 0])  # exactly k = 4 steps late


def test_long_chunk_teacher_gives_wide_latents(data_dir, tmp_path):
    """EXP-0002: a 40-step teacher's chunk latent is 40 x 7 = 280 wide, aligned with the data rows."""
    from ecvla.teacher import precompute

    teacher_dir = tmp_path / "teacher40"
    train(parser().parse_args(["--data", str(data_dir), "--suites", "libero_fake", "--steps", "20", "--batch", "4",
                               "--log-every", "10", "--val-per-task", "1", "--device", "cpu", "--condition", "task-id",
                               "--size", "20000", "--chunk", "40", "--out", str(teacher_dir)]))
    assert torch.load(teacher_dir / "model.pt", weights_only=False)["info"]["policy"]["chunk"] == 40
    precompute(str(teacher_dir), str(data_dir / "libero_fake"), str(tmp_path / "z/libero_fake"))
    z = np.load(tmp_path / "z/libero_fake/a_demo.npz")["z_chunk"]
    assert z.shape == (np.load(data_dir / "libero_fake/a_demo.npz")["ep_start"][-1], 280)


def test_track_student_trains_on_the_aligned_window_and_runs_closed_loop(data_dir, tmp_path):
    """EXP-0003: a `track` student on a 40-step teacher's plan gets the 8-step window due at its age (training and
    closed loop), starts as the plan, and the replay runner executes plan step `age`."""
    from ecvla.evaluate import ReplayRunner, Runner
    from ecvla.policy import plan_window
    from ecvla.teacher import Teacher, precompute
    from ecvla.train import Batcher

    common = ["--data", str(data_dir), "--suites", "libero_fake", "--batch", "4", "--log-every", "20",
              "--val-per-task", "1", "--device", "cpu"]
    teacher_dir = tmp_path / "teacher40"
    train(parser().parse_args(common + ["--steps", "20", "--condition", "task-id", "--size", "20000", "--chunk", "40",
                                        "--out", str(teacher_dir)]))
    precompute(str(teacher_dir), str(data_dir / "libero_fake"), str(tmp_path / "z/libero_fake"))

    data = EpisodeData(data_dir, ["libero_fake"], val_per_task=1, latent_dir=tmp_path / "z")
    stats = {k: v for k, v in data.norm_stats().items() if k != "latent"}
    bt = Batcher(data, stats, "z-teacher", torch.device("cpu"), 8, track=True)
    ep = data.split(False)[:2]
    b, _, _ = bt(ep, np.random.default_rng(0))
    assert b["latent"].shape[-1] == 56
    i, t = 1, 25  # check one step against a direct gather from the stored rows
    age = int(round(float(b["age"][i, t]) * 20))
    src = ep[i][0] + t - age
    want = plan_window(torch.from_numpy(data.rows["latent"][src]), torch.tensor(age), 8)
    torch.testing.assert_close(b["latent"][i, t], want)

    student = tmp_path / "student"
    train(parser().parse_args(common + ["--steps", "40", "--condition", "z-teacher", "--size", "10000", "--track",
                                        "--out", str(student), "--latent-dir", str(tmp_path / "z"),
                                        "--teacher", str(teacher_dir)]))
    ckpt = torch.load(student / "model.pt", weights_only=False)
    pol = ckpt["info"]["policy"]
    assert pol["track"] and pol["latent_dim"] == 56 and pol["track_bound"][6] == 0 and "latent" not in ckpt["stats"]

    obs = {"proprio": np.zeros(PROPRIO_DIM), "torque": np.zeros(TORQUE_DIM), "ft": np.zeros(FT_DIM), "fast_valid": 0.0,
           "obj_pose": np.zeros((3, 7)), "contact": np.zeros(3), "fix_state": np.zeros((2, 2))}
    teacher = Teacher(teacher_dir)
    r = Runner(ckpt, "b")
    r.reset(40, np.random.default_rng(0), k=20, teacher=teacher)
    for _ in range(25):
        a = r.act(obs)
    assert np.isfinite(a).all() and r.src[24] == 0

    rep = ReplayRunner(teacher, "b")
    rep.reset(40, np.random.default_rng(0), k=20)
    acts = [rep.act(obs) for _ in range(25)]
    m, s = teacher.stats["actions"]
    plan0 = rep.online.history[0].reshape(40, 7)
    np.testing.assert_allclose(acts[24], plan0[24] * s + m, rtol=1e-5)  # step 24 executes plan(0)[age 24]
    np.testing.assert_allclose(acts[5], rep.online.history[5].reshape(40, 7)[0] * s + m, rtol=1e-5)  # no plan yet

    # with a zero residual, the student executes exactly what replay executes once a plan has arrived
    np.testing.assert_allclose(ckpt["stats"]["actions"], teacher.stats["actions"], rtol=1e-6)
    sd = {k: (torch.zeros_like(v) if k.startswith("head.") else v) for k, v in ckpt["state_dict"].items()}
    r = Runner(dict(ckpt, state_dict=sd), "b")
    r.reset(40, np.random.default_rng(0), k=20, teacher=teacher)
    student = [r.act(obs) for _ in range(40)]
    np.testing.assert_allclose(np.array(student[20:]), np.array(acts[20:] + [rep.act(obs) for _ in range(15)]),
                               rtol=1e-5, atol=1e-6)
