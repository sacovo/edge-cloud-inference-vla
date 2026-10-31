"""EXP-0004 §6 check: is the scene static during a hold? After LIBERO's settle steps (as evaluate.py), hold the robot
with no-op actions for HOLD steps and report the largest end-effector and object displacement (and EEF rotation).

Usage: python scripts/exp0004_hold_drift.py [--episodes 5] [--hold 20] [--out drift.json]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ecvla.evaluate import SETTLE_STEPS
from ecvla.libero_env import make_env
from ecvla.replay import object_names
from ecvla.sensors import NOOP_ACTION

SUITES = ("libero_spatial", "libero_object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path

    rows = []
    for suite in SUITES:
        bm = benchmark.get_benchmark_dict()[suite]()
        for task_id in range(10):
            task = bm.get_task(task_id)
            env = make_env(os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                           cameras=False)
            init = bm.get_task_init_states(task_id)
            for ep in range(a.episodes):
                env.reset()
                obs = env.set_init_state(init[ep])
                for _ in range(SETTLE_STEPS):
                    obs, _, _, _ = env.step(NOOP_ACTION)
                objs = object_names(obs)
                e0, q0, p0 = obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy(), {o: obs[f"{o}_pos"].copy() for o in objs}
                for _ in range(a.hold):
                    obs, _, _, _ = env.step(NOOP_ACTION)
                dq = abs(float(np.dot(q0, obs["robot0_eef_quat"])))
                rows.append({"suite": suite, "task": task_id, "episode": ep,
                             "eef_mm": 1000 * float(np.linalg.norm(obs["robot0_eef_pos"] - e0)),
                             "eef_rot_deg": float(np.degrees(2 * np.arccos(min(1.0, dq)))),
                             "obj_mm_max": 1000 * max(float(np.linalg.norm(obs[f"{o}_pos"] - p0[o])) for o in objs)})
            env.close()
            print(suite, task_id, " ".join(f"{k}={max(r[k] for r in rows[-a.episodes:]):.2f}"
                                           for k in ("eef_mm", "eef_rot_deg", "obj_mm_max")), flush=True)
    summ = {k: {"max": max(r[k] for r in rows), "median": float(np.median([r[k] for r in rows]))}
            for k in ("eef_mm", "eef_rot_deg", "obj_mm_max")}
    print(json.dumps(summ))
    if a.out:
        Path(a.out).write_text(json.dumps({"hold": a.hold, "settle": SETTLE_STEPS, "summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
