"""Online sensor frontend (closed loop) against robosuite's own observations and the offline window builder."""

from pathlib import Path

import h5py
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / (
    "data/libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
pytestmark = pytest.mark.skipif(not SRC.exists(), reason=f"needs LIBERO source file {SRC}")


def test_online_sensors_match_robot_obs_and_offline_windows():
    from ecvla.libero_env import bddl_path, make_env
    from ecvla.replay import SubstepLogger, object_names
    from ecvla.sensors import OnlineSensors, fast_windows

    with h5py.File(SRC) as f:
        env = make_env(bddl_path(f["data"].attrs["bddl_file_name"]), cameras=False)
        states, actions = f["data/demo_0/states"][()], f["data/demo_0/actions"][()]
    env.reset()
    logger = SubstepLogger(env)
    obs = env.set_init_state(states[0])
    objs = object_names(obs)
    sensors = OnlineSensors(env, logger, objs)
    logger.take()
    seen, subs = [], []
    for t in range(30):
        o = sensors.observe(obs)
        # index mapping (robosuite's observables lag the sim state slightly after a step, so not exact)
        np.testing.assert_allclose(o["proprio"][:7], obs["robot0_joint_pos"], atol=2e-3)
        np.testing.assert_allclose(o["proprio"][7:14], obs["robot0_joint_vel"], atol=5e-2)
        np.testing.assert_allclose(o["proprio"][14:], obs["robot0_gripper_qpos"], atol=2e-3)
        seen.append(o)
        obs, *_ = env.step(actions[t])
        subs.append({k: v.copy() for k, v in logger.buf.items()})  # peek: observe() takes them next step
    sub = {k: np.concatenate([np.asarray(s[k]) for s in subs]) for k in ("joint_torque", "ee_force", "ee_torque")}
    torque, ft = fast_windows(sub, 30)
    np.testing.assert_allclose(np.stack([o["torque"] for o in seen]), torque, atol=1e-6)
    np.testing.assert_allclose(np.stack([o["ft"] for o in seen]), ft, atol=1e-6)
    assert [o["fast_valid"] for o in seen[:2]] == [0, 1]
    env.close()
