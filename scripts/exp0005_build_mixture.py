"""EXP-0005: build the FAST fine-tuning mixture as one LeRobot dataset (README §3 recipe).

Per source dataset: native action and state formats, normalized per dimension to [-1, 1] by that dataset's own
1% / 99% quantiles (clipped), zero-padded to the mixture's widest state / action; both cameras at native resolution
(pi0_fast resizes with padding); the task text carries the condition, e.g.
  "pick up the bowl; robot: franka; control: delta end effector; 20 Hz".
LeRobot's normalizer is then identity (multi-dataset stats would be merged into one set otherwise). Per-source
quantiles, dims and conventions go to <out>/mixture_stats.json (needed to de-normalize at evaluation).

Sources (`SOURCES`): LIBERO raw HDF5s (openpi conventions: 180° images, state = eef pos + axis-angle + gripper qpos)
and LeRobot hub ports of OXE datasets. Before trusting a source, check the printed gripper diagnostics: the gripper
action is aligned to LIBERO's convention (-1 open, +1 close) via `grip_sign`.

Runs in the VLM env:  uv run --project vlm python scripts/exp0005_build_mixture.py --sources libero utaustin_mutex \
                        --out $ECVLA_SCRATCH/exp0005/datasets/mix_debug [--max-episodes 50]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
PROMPT = "{condition}; task: {task}"  # condition first: pi0_fast cuts the prompt at 200 tokens (long OXE instructions)
FPS = 20  # one container rate; all sources so far are 20 Hz (mixed rates: open design question, README §9)
IMG = ("observation.images.image", "observation.images.wrist_image")
SOURCES = {
    # name: loader, condition fields, gripper sign (multiplies the normalized gripper action), rate
    "libero": {"loader": "libero", "suites": ("libero_spatial", "libero_object"), "robot": "franka",
               "control": "delta end effector", "hz": 20, "grip_dim": 6, "grip_sign": 1},
    # mutex gripper: 0/1 with 1 = open in the LeRobot port (debug build 2026-11-03: commanded 1 goes with a wide
    # finger opening, corr 0.67 with state dim 7 = width), so flipped to LIBERO's +1 = close. Octo inverts the raw
    # RLDS gripper to its own 1 = open (oxe_standardization_transforms.py L781-796), implying raw 1 = close: the
    # LeRobot port may differ from the RLDS original (not checked). Trust the data of the copy actually used.
    "utaustin_mutex": {"loader": "lerobot", "repo": "lerobot/utaustin_mutex", "robot": "franka",
                       "control": "delta end effector", "hz": 20, "grip_dim": 6, "grip_sign": -1},
}


def libero_episodes(cfg, max_eps):
    files = sorted(f for su in cfg["suites"] for f in glob.glob(str(S / "raw" / su / "*.hdf5")))
    n = 0
    for f in files:
        with h5py.File(f, "r") as h:
            task = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
            for name in sorted(h["data"], key=lambda s: int(s.split("_")[1])):
                if max_eps and n >= max_eps:
                    return
                d = h["data"][name]
                yield {
                    "task": task,
                    "state": np.concatenate([d["obs/ee_pos"][:], d["obs/ee_ori"][:], d["obs/gripper_states"][:]], 1),
                    "action": d["actions"][:].astype(np.float32),
                    IMG[0]: np.ascontiguousarray(d["obs/agentview_rgb"][:][:, ::-1, ::-1]),  # 180°, as openpi
                    IMG[1]: np.ascontiguousarray(d["obs/eye_in_hand_rgb"][:][:, ::-1, ::-1]),
                }
                n += 1


def lerobot_episodes(cfg, max_eps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_ds = LeRobotDataset(cfg["repo"], download_videos=False)
    n_eps = meta_ds.meta.total_episodes if not max_eps else min(max_eps, meta_ds.meta.total_episodes)
    ds = LeRobotDataset(cfg["repo"], episodes=list(range(n_eps)))
    ep = ds.hf_dataset["episode_index"]
    ep = np.asarray([int(e) for e in ep])
    for e in range(n_eps):
        idx = np.nonzero(ep == e)[0]
        frames = [ds[int(i)] for i in idx]
        to_img = lambda x: (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # noqa: E731
        yield {
            "task": frames[0]["task"],
            "state": np.stack([f["observation.state"].numpy() for f in frames]).astype(np.float32),
            "action": np.stack([f["action"].numpy() for f in frames]).astype(np.float32),
            IMG[0]: np.stack([to_img(f[IMG[0]]) for f in frames]),
            IMG[1]: np.stack([to_img(f[IMG[1]]) for f in frames]),
        }


def clean_task(t):
    """LeRobot OXE ports carry RLDS artifacts in the task text: `tf.Tensor(b"...", shape=(), dtype=string)` wrappers
    and literal backslash-n sequences (seen in utaustin_mutex)."""
    t = t.strip()
    if t.startswith('tf.Tensor(b"') or t.startswith("tf.Tensor(b'"):
        t = t[len('tf.Tensor(b"'):]
        t = t[: t.rfind(", shape=")].rstrip("\"'")
    return " ".join(t.replace("\\n", " ").replace("\n", " ").split())


def quantiles(x):
    return np.quantile(x, 0.01, 0), np.quantile(x, 0.99, 0)


def normalize(x, q):
    lo, hi = q
    return np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -1, 1).astype(np.float32)


def gripper_diagnostics(eps, cfg):
    """Gripper action values and their correlation with the next-step change of each state dim (sign check)."""
    a = np.concatenate([e["action"][:-1, cfg["grip_dim"]] for e in eps])
    ds = np.concatenate([np.diff(e["state"], axis=0) for e in eps])
    corr = [float(np.corrcoef(a, ds[:, j])[0, 1]) if ds[:, j].std() > 0 else 0.0 for j in range(ds.shape[1])]
    vals, counts = np.unique(np.round(a, 2), return_counts=True)
    top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:6]
    return {"grip_action_top_values": [[v, c] for c, v in top], "corr_with_next_state_change": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, choices=list(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0, help="per source; 0 = all")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = {}
    for name in a.sources:
        cfg = SOURCES[name]
        gen = libero_episodes if cfg["loader"] == "libero" else lerobot_episodes
        loaded[name] = list(gen(cfg, a.max_episodes))
        print(f"{name}: {len(loaded[name])} episodes, {sum(len(e['action']) for e in loaded[name])} frames", flush=True)
    sd = max(e["state"].shape[1] for eps in loaded.values() for e in eps)
    ad = max(e["action"].shape[1] for eps in loaded.values() for e in eps)
    h, w = next(iter(loaded.values()))[0][IMG[0]].shape[1:3]
    stats = {"fps": FPS, "state_dim": sd, "action_dim": ad, "prompt_format": PROMPT, "sources": {}}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    features = {
        "observation.state": {"dtype": "float32", "shape": (sd,), "names": None},
        "action": {"dtype": "float32", "shape": (ad,), "names": None},
        **{k: {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]} for k in IMG},
    }
    out = Path(a.out)
    ds = LeRobotDataset.create(repo_id=f"local/{out.name}", root=out, fps=FPS, features=features,
                               robot_type="mixed", use_videos=False)
    for name, eps in loaded.items():
        cfg = SOURCES[name]
        assert all(e[IMG[0]].shape[1:3] == (h, w) for e in eps), f"{name}: image size differs from {(h, w)}"
        qs = quantiles(np.concatenate([e["state"] for e in eps]))
        qa = quantiles(np.concatenate([e["action"] for e in eps]))
        cond = f"robot: {cfg['robot']}; control: {cfg['control']}; {cfg['hz']} Hz"
        texts = {e["task"]: PROMPT.format(condition=cond, task=clean_task(e["task"])) for e in eps}
        # pi0_fast's prompt adds "Task: ..., State: <state_dim numbers>;" around this; report the task part's length
        n_tok = max(len(tok(t)["input_ids"]) for t in texts.values())
        stats["sources"][name] = {
            "condition": cond, "episodes": len(eps), "frames": int(sum(len(e["action"]) for e in eps)),
            "state_dim": int(eps[0]["state"].shape[1]), "action_dim": int(eps[0]["action"].shape[1]),
            "state_q01": qs[0].tolist(), "state_q99": qs[1].tolist(),
            "action_q01": qa[0].tolist(), "action_q99": qa[1].tolist(), "grip_sign": cfg["grip_sign"],
            "gripper": gripper_diagnostics(eps, cfg), "max_task_tokens": n_tok,
            "example_prompt": next(iter(texts.values())),
        }
        print(name, json.dumps(stats["sources"][name]["gripper"]), flush=True)
        for e in eps:
            st = np.zeros((len(e["state"]), sd), np.float32)
            st[:, : e["state"].shape[1]] = normalize(e["state"], qs)
            act = np.zeros((len(e["action"]), ad), np.float32)
            act[:, : e["action"].shape[1]] = normalize(e["action"], qa)
            act[:, cfg["grip_dim"]] *= cfg["grip_sign"]
            for t in range(len(act)):
                ds.add_frame({"observation.state": st[t], "action": act[t], IMG[0]: e[IMG[0]][t],
                              IMG[1]: e[IMG[1]][t], "task": texts[e["task"]]})
            ds.save_episode()
    ds.finalize()
    (out / "mixture_stats.json").write_text(json.dumps(stats, indent=1))
    print("done", out, {k: (v["episodes"], v["frames"]) for k, v in stats["sources"].items()})


if __name__ == "__main__":
    main()
