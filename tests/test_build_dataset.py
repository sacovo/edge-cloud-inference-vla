"""Contract of the Phase A dataset file that training code reads (see src/ecvla/build_dataset.py docstring)."""

from pathlib import Path

import h5py
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / (
    "data/libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
pytestmark = pytest.mark.skipif(not SRC.exists(), reason=f"needs LIBERO source file {SRC}")


def test_build_writes_aligned_multirate_demo(tmp_path):
    from ecvla.build_dataset import build, decode_jpeg

    out = tmp_path / "task.h5"
    build(str(SRC), str(out), image_size=128, max_demos=1)
    assert out.exists() and not out.with_suffix(".h5.tmp").exists()

    with h5py.File(out) as f, h5py.File(SRC) as src:
        g, s = f["demo_0"], src["data/demo_0"]
        T = len(s["actions"])
        # 20 Hz streams: one row per control step
        assert g["object_pose"].shape[0] == T and g["actions"].shape == (T, 7)
        np.testing.assert_array_equal(g["sim_state"][()], s["states"][()])
        # 500 Hz streams: 25 physics substeps of 2 ms per 50 ms control step
        sub_t = g["substeps/time"][()]
        assert len(sub_t) == 25 * T
        np.testing.assert_allclose(np.diff(sub_t), 0.002, atol=1e-5)
        for k in ("joint_torque", "joint_pos", "joint_vel", "ee_force", "ee_torque"):
            assert g[f"substeps/{k}"].shape[0] == 25 * T
        # images rendered from state t match LIBERO's own recorded obs t (independent renderer, JPEG-compressed)
        ours = np.stack([decode_jpeg(x) for x in g["images/agentview"][()]]).astype(float)
        assert ours.shape == (T, 128, 128, 3)
        assert np.abs(ours - s["obs/agentview_rgb"][()].astype(float)).mean() < 15


def test_build_skips_existing_output(tmp_path):
    from ecvla.build_dataset import build

    out = tmp_path / "task.h5"
    out.write_bytes(b"finished")
    assert build(str(SRC), str(out), max_demos=1) == {"skipped": True}
    assert out.read_bytes() == b"finished"
