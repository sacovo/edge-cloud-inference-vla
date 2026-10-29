"""Closed-loop evaluation in a real LIBERO scene (needs the libero extra; no dataset files)."""

import json

import pytest

from tests.test_train import fake_task

pytest.importorskip("libero")
SPATIAL_0 = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
SPATIAL_0_OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_2", "cookies_1", "glazed_rim_porcelain_ramekin_1", "plate_1"]


def test_model_without_fixture_data_runs_in_scene_with_fixtures(tmp_path):
    """Regression: v1 data (no fixture state) gives a model without fixture tokens; the scene's drawers must not matter."""
    from ecvla.evaluate import evaluate
    from ecvla.train import parser, train

    fake_task(tmp_path / "data/libero_spatial/t0_demo.npz", SPATIAL_0, SPATIAL_0_OBJECTS, n_demos=4)
    run = tmp_path / "run"
    train(parser().parse_args(["--data", str(tmp_path / "data"), "--suites", "libero_spatial", "--condition",
                               "oracle-fresh", "--size", "10000", "--out", str(run), "--steps", "5", "--batch", "2",
                               "--val-per-task", "1", "--log-every", "5", "--device", "cpu"]))
    res = evaluate(str(run), "libero_spatial", 0, str(tmp_path / "eval.json"), episodes=1)
    assert res["episodes"] == 1 and res["per_episode"][0]["steps"] > 0
    assert json.loads((tmp_path / "eval.json").read_text())["task"] == SPATIAL_0
