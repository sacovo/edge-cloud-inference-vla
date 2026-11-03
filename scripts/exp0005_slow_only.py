"""EXP-0005 `slow-only` check: the fine-tuned `pi0_fast` acts alone in the LIBERO sim, executing its decoded 1 s chunks.

Inputs built as in training (`exp0005_build_mixture.py`, openpi's LIBERO conventions): both cameras rotated 180° at
128 px (the policy resizes), state = eef pos + axis-angle (openpi's `_quat2axisangle`, robosuite's formula) + gripper
qpos, normalized with LIBERO's quantiles from `mixture_stats.json`; task text = instruction + the LIBERO condition.
Actions are de-normalized with LIBERO's action quantiles. The policy replans when its 20-step queue is empty
(`n_action_steps` = chunk), i.e. open loop within each 1 s chunk. Protocol otherwise as `ecvla.evaluate`: LIBERO's
fixed initial states, 10 settle no-ops, the suite's step limit.

Before the episodes, a consistency check replays a few states of a recorded demo of the same task and compares the
state built here with the HDF5's stored `ee_pos` / `ee_ori` / `gripper_states` (what the policy was trained on).

Runs in the VLM env with the project on the path (GPU node):
  PYTHONPATH=src uv run --project vlm python scripts/exp0005_slow_only.py --ckpt <checkpoints/N/pretrained_model> \
    --mixture <mixture dir> --suite libero_object --task 0 --episodes 2 --out <json>
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ecvla.evaluate import MAX_STEPS, SETTLE_STEPS
from ecvla.libero_env import make_env
from ecvla.sensors import NOOP_ACTION

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
IMG = ("observation.images.image", "observation.images.wrist_image")


def quat2axisangle(quat):
    """openpi `examples/libero/main.py` `_quat2axisangle` (robosuite transform_utils); quat = (x, y, z, w)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def raw_state(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])


def consistency_check(env, task_key, suite):
    """Max |difference| between the state built from sim obs and the HDF5 obs, on replayed demo states."""
    f = S / "raw" / suite / f"{task_key}_demo.hdf5"
    with h5py.File(f, "r") as h:
        d = h["data/demo_0"]
        ts = [0, len(d["states"]) // 2, len(d["states"]) - 1]
        diffs = []
        for t in ts:
            obs = env.set_init_state(d["states"][t])
            ref = np.concatenate([d["obs/ee_pos"][t], d["obs/ee_ori"][t], d["obs/gripper_states"][t]])
            diffs.append(np.abs(raw_state(obs) - ref))
    d = np.max(diffs, axis=0)
    return {"file": str(f), "frames": ts, "max_abs_pos": float(d[:3].max()), "max_abs_axisangle": float(d[3:6].max()),
            "max_abs_gripper": float(d[6:].max())}


def load_policy(ckpt):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from peft import PeftConfig, PeftModel

    base_path = PeftConfig.from_pretrained(ckpt).base_model_name_or_path
    base = PI0FastPolicy.from_pretrained(base_path)
    head = base.model.paligemma_with_expert.paligemma.lm_head.weight.detach().clone()
    peft = PeftModel.from_pretrained(base, ckpt)
    trained_head = peft.base_model.model.model.paligemma_with_expert.paligemma.lm_head
    w = getattr(trained_head, "modules_to_save", {}).get("default", trained_head).weight
    head_changed = float((w.detach().cpu() - head.cpu()).abs().max())
    policy = peft.base_model.model.to("cuda").eval()  # LoRA layers are injected in place
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    return policy, pre, post, {"base": base_path, "lm_head_max_change": head_changed}


def to_img(x):
    img = np.ascontiguousarray(x[::-1, ::-1])  # 180°, as in training (openpi)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--task-key", help="task by its bddl name (overrides --task)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path

    mix = json.loads((Path(a.mixture) / "mixture_stats.json").read_text())
    stats = mix["sources"]["libero"]
    slo, shi = np.array(stats["state_q01"]), np.array(stats["state_q99"])
    alo, ahi = np.array(stats["action_q01"]), np.array(stats["action_q99"])
    bm = benchmark.get_benchmark_dict()[a.suite]()
    if a.task_key:
        a.task = [bm.get_task(i).bddl_file.removesuffix(".bddl") for i in range(bm.n_tasks)].index(a.task_key)
    task = bm.get_task(a.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    key = Path(bddl).stem
    fmt = mix.get("prompt_format", "{task}; {condition}")  # mixtures built before 2026-11-03 put the condition last
    text = fmt.format(condition=stats["condition"], task=task.language)
    env = make_env(bddl, image_size=128, cameras=True)
    out = {"ckpt": a.ckpt, "suite": a.suite, "task_id": a.task, "task": key, "prompt": text,
           "consistency": consistency_check(env, key, a.suite)}
    print("consistency:", out["consistency"], flush=True)
    policy, pre, post, out["policy"] = load_policy(a.ckpt)
    print("policy:", out["policy"], flush=True)
    init = bm.get_task_init_states(a.task)
    eps = []
    for ep in range(a.episodes):
        env.reset()
        obs = env.set_init_state(init[ep])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(NOOP_ACTION)
        policy.reset()
        success, steps, chunk_s, acts = False, 0, [], []
        for steps in range(1, MAX_STEPS[a.suite] + 1):
            st = np.clip(2 * (raw_state(obs) - slo) / np.maximum(shi - slo, 1e-6) - 1, -1, 1)
            batch = {IMG[0]: to_img(obs["agentview_image"]), IMG[1]: to_img(obs["robot0_eye_in_hand_image"]),
                     "observation.state": torch.from_numpy(st).float()[None], "task": [text]}
            new_chunk, t0 = len(policy._action_queue) == 0, time.time()
            with torch.no_grad():
                act = post(policy.select_action(pre(batch)))
            if new_chunk:
                chunk_s.append(time.time() - t0)
            an = act[0].float().cpu().numpy()[:7]
            raw = alo + (an + 1) / 2 * (ahi - alo)
            acts.append(raw)
            obs, _, done, _ = env.step(np.clip(raw, -1, 1))
            if done or env.check_success():
                success = True
                break
        acts = np.array(acts)
        eps.append({"episode": ep, "success": success, "steps": steps, "chunks": len(chunk_s),
                    "chunk_decode_s_mean": float(np.mean(chunk_s)) if chunk_s else None,
                    "action_mean": acts.mean(0).round(4).tolist(), "action_std": acts.std(0).round(4).tolist()})
        print(eps[-1], flush=True)
    env.close()
    out["episodes"] = eps
    out["success_rate"] = float(np.mean([e["success"] for e in eps]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("done", a.out, out["success_rate"])


if __name__ == "__main__":
    main()
