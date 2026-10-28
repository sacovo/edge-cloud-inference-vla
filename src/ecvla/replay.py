"""Replay LIBERO demonstrations to regenerate multi-rate sensor streams (EXP-0001 A0).

Two modes:
- "hybrid": before each control step, reset the sim to the recorded state, apply the recorded action, and log
  every physics substep (500 Hz). The trajectory follows the demo exactly; the logged torque and F/T come from
  actually executing the action. Reports how far each simulated step lands from the next recorded state.
- "open_loop": reset once to the initial state and execute all recorded actions. Reports task success.
"""

import numpy as np

SUBSTEP_KEYS = ("time", "joint_torque", "joint_pos", "joint_vel", "ee_force", "ee_torque")


class SubstepLogger:
    """Wraps sim.step to record robot torque, joint state and end-effector F/T at every physics substep."""

    def __init__(self, env):
        self.sim = env.sim
        robot = env.robots[0]
        self.qpos_idx = np.asarray(robot._ref_joint_pos_indexes)
        self.qvel_idx = np.asarray(robot._ref_joint_vel_indexes)
        m = self.sim.model
        self.force_adr = m.sensor_adr[m.sensor_name2id("gripper0_force_ee")]
        self.torque_adr = m.sensor_adr[m.sensor_name2id("gripper0_torque_ee")]
        self.buf = {k: [] for k in SUBSTEP_KEYS}
        self._orig_step = self.sim.step
        self.sim.step = self._step

    def _step(self, *args, **kwargs):
        self._orig_step(*args, **kwargs)
        d = self.sim.data
        self.buf["time"].append(d.time)
        self.buf["joint_torque"].append(d.qfrc_actuator[self.qvel_idx].copy())
        self.buf["joint_pos"].append(d.qpos[self.qpos_idx].copy())
        self.buf["joint_vel"].append(d.qvel[self.qvel_idx].copy())
        self.buf["ee_force"].append(d.sensordata[self.force_adr : self.force_adr + 3].copy())
        self.buf["ee_torque"].append(d.sensordata[self.torque_adr : self.torque_adr + 3].copy())

    def take(self) -> dict:
        out = {k: np.asarray(v) for k, v in self.buf.items()}
        self.buf = {k: [] for k in SUBSTEP_KEYS}
        return out

    def detach(self):
        self.sim.step = self._orig_step


def object_names(obs: dict) -> list[str]:
    return sorted(k[: -len("_pos")] for k in obs if k.endswith("_pos") and not k.startswith("robot0") and "_to_" not in k)


def replay_hybrid(env, states: np.ndarray, actions: np.ndarray) -> dict:
    """Replay a demo by resetting to each recorded state; log 500 Hz substeps and 20 Hz object poses."""
    env.reset()
    log = SubstepLogger(env)
    obs = env.set_init_state(states[0])
    log.take()
    objs = object_names(obs)
    obj_pose, sub, step_err = [], [], []
    for t in range(len(actions)):
        obs = env.regenerate_obs_from_state(states[t])
        obj_pose.append(np.concatenate([np.r_[obs[f"{o}_pos"], obs[f"{o}_quat"]] for o in objs]))
        log.take()
        env.step(actions[t])
        sub.append(log.take())
        if t + 1 < len(states):
            # compare positions only (state = [time, qpos, qvel]); skip time
            nq = env.sim.model.nq
            step_err.append(np.abs(env.get_sim_state()[1 : 1 + nq] - states[t + 1][1 : 1 + nq]).max())
    log.detach()
    substeps = {k: np.concatenate([s[k] for s in sub]) for k in SUBSTEP_KEYS}
    return {
        "objects": objs,
        "object_pose": np.asarray(obj_pose),
        "substeps": substeps,
        "substeps_per_step": np.array([len(s["time"]) for s in sub]),
        "step_qpos_err": np.asarray(step_err),
        "success_at_end": bool(env.check_success()),
    }


def replay_open_loop(env, states: np.ndarray, actions: np.ndarray) -> dict:
    env.reset()
    env.set_init_state(states[0])
    nq = env.sim.model.nq
    success_step, drift = None, []
    for t, a in enumerate(actions):
        env.step(a)
        if t + 1 < len(states):
            drift.append(np.abs(env.get_sim_state()[1 : 1 + nq] - states[t + 1][1 : 1 + nq]).max())
        if success_step is None and env.check_success():
            success_step = t
    return {"success": success_step is not None, "success_step": success_step, "qpos_drift": np.asarray(drift)}
