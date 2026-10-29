"""Fast-policy sensor inputs at their EXP-0001 rates, built the same way from the Phase A dataset and from a live env.

Observation t is the sim state at the start of control step t (20 Hz). The policy sees:
- proprio (20 Hz): joint pos (7), joint vel (7), gripper qpos (2), read from the state at t
- torque window (60 Hz): the 3 joint-torque samples of control step t-1 (substeps TORQUE_SUB of its 25)
- F/T window (100 Hz): the 5 wrist force/torque samples of control step t-1 (substeps FT_SUB)
- fast_valid: 0 when there is no previous step (t=0) or that step was physically unstable in replay
Object state (oracle conditions) is read at t: per object [pos(3), quat(4)] plus a gripper-contact flag.
"""

import numpy as np

SUBSTEPS = 25  # 500 Hz physics / 20 Hz control
TORQUE_SUB = np.array([7, 15, 24])  # last substep at or before k/60 s within the step
FT_SUB = np.array([4, 9, 14, 19, 24])
N_JOINTS = 7
PROPRIO_DIM = 2 * N_JOINTS + 2
TORQUE_DIM = len(TORQUE_SUB) * N_JOINTS
FT_DIM = len(FT_SUB) * 6
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def proprio_from_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Panda joints are qpos/qvel[0:7] and the gripper fingers qpos[7:9] in every LIBERO scene (checked in tests)."""
    return np.concatenate([qpos[..., :N_JOINTS], qvel[..., :N_JOINTS], qpos[..., N_JOINTS : N_JOINTS + 2]], axis=-1)


def fast_windows(substeps: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, TORQUE_DIM), (T, FT_DIM) windows for observations 0..T-1 from 500 Hz substeps of T control steps.

    Row t holds step t-1's samples; row 0 is zero (no previous step).
    """
    tq = substeps["joint_torque"].reshape(n_steps, SUBSTEPS, N_JOINTS)[:, TORQUE_SUB]
    ft = np.concatenate([substeps["ee_force"], substeps["ee_torque"]], axis=-1).reshape(n_steps, SUBSTEPS, 6)[:, FT_SUB]
    torque = np.zeros((n_steps, TORQUE_DIM), np.float32)
    force = np.zeros((n_steps, FT_DIM), np.float32)
    torque[1:] = tq[:-1].reshape(n_steps - 1, -1)
    force[1:] = ft[:-1].reshape(n_steps - 1, -1)
    return torque, force


def gripper_contacts(env, objects: list[str]) -> np.ndarray:
    """1 where the gripper touches the object; the sim must be forwarded to the current state."""
    inner = env.env
    gripper = inner.robots[0].gripper
    return np.array([inner.check_contact(gripper, inner.get_object(o)) for o in objects], dtype=np.float32)


def object_pose(obs: dict, objects: list[str]) -> np.ndarray:
    return np.stack([np.r_[obs[f"{o}_pos"], obs[f"{o}_quat"]] for o in objects]).astype(np.float32)


class OnlineSensors:
    """Builds the same per-step inputs as the prepared dataset while the policy runs in the env.

    Call `observe(obs)` at each control step before acting; `env.step` must run with a SubstepLogger attached.
    """

    def __init__(self, env, logger, objects: list[str]):
        self.env, self.logger, self.objects = env, logger, objects
        self.first = True

    def observe(self, obs: dict) -> dict:
        sim = self.env.sim
        sim.forward()  # contacts of the current state, not of the last substep's pre-integration state
        sub = self.logger.take()
        torque = np.zeros(TORQUE_DIM, np.float32)
        force = np.zeros(FT_DIM, np.float32)
        if not self.first:
            assert len(sub["time"]) == SUBSTEPS, len(sub["time"])
            torque = sub["joint_torque"][TORQUE_SUB].reshape(-1)
            force = np.concatenate([sub["ee_force"], sub["ee_torque"]], axis=-1)[FT_SUB].reshape(-1)
        out = {
            "proprio": proprio_from_state(sim.data.qpos, sim.data.qvel).astype(np.float32),
            "torque": torque.astype(np.float32),
            "ft": force.astype(np.float32),
            "fast_valid": np.float32(not self.first),
            "obj_pose": object_pose(obs, self.objects),
            "contact": gripper_contacts(self.env, self.objects),
        }
        self.first = False
        return out
