"""EXP-0005: build the FAST fine-tuning mixture as one LeRobot dataset (README §3 recipe).

Per source: native action and state formats, each dimension normalized to [-1, 1] by that source's own 1% / 99%
quantiles (clipped), zero-padded to the mixture's widest state / action; main + wrist camera resized with padding to
224 px and stored as video; the task text carries the condition first (pi0_fast cuts prompts at 200 tokens), e.g.
  "robot: franka; control: delta end effector; 20 Hz; task: pick up the bowl".
Each source stays at its native rate (a chunk is counted in steps; the rate is in the prompt). LeRobot's normalizer is
then identity. Per-source quantiles, signs and dims go to <out>/mixture_stats.json (needed to de-normalize).

LIBERO is re-rendered in our sim from the recorded states (`exp0005_common`, also used by the slow-only check): the
recorded HDF5 obs are one step after the state the action was taken in, and our sim renders Object scenes darker and
with robosuite's gripper-site marker, so recorded images would differ from what the policy sees in closed loop.
OXE sources come from their LeRobot hub ports; each episode's video segment is decoded in one pass (pyav).

Gripper sign: chosen so that an episode's first gripper command (the gripper starts open) maps to LIBERO's -1 (open);
`--check` prints the evidence per source before anything is written.

Modes (VLM env; the full build needs EGL for LIBERO, i.e. a GPU node):
  uv run --project vlm python scripts/exp0005_build_mixture.py --check --sources ... --out <dir>   (stats only)
  PYTHONPATH=src uv run --project vlm python scripts/exp0005_build_mixture.py --sources ... --out <dir>
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp0005_common import IMG, IMG_SIZE, libero_images, libero_state, resize_pad  # noqa: E402

S = Path(os.environ.get("ECVLA_SCRATCH", "/scratch/sandro.covo/ecvla"))
PROMPT = "{condition}; task: {task}"
FPS = 20  # container rate only; each source keeps its native steps (README §3)
DROID_FILE_STRIDE = 20  # every 20th video file of DROID (~300 episodes each) -> ~5k episodes
EEF = "delta end effector"
SKIPPED: dict[str, list[int]] = {}  # per source: episodes whose video segment cannot be decoded (port data errors)
SOURCES = {
    # name: loader, cameras (main, wrist), condition fields, gripper action dim
    "libero": {"loader": "libero", "suites": ("libero_spatial", "libero_object"), "robot": "franka", "control": EEF,
               "hz": 20, "grip_dim": 6},
    "utaustin_mutex": {"repo": "lerobot/utaustin_mutex", "cams": ("image", "wrist_image"), "robot": "franka",
                       "control": EEF, "hz": 20, "grip_dim": 6},
    "austin_sirius": {"repo": "lerobot/austin_sirius_dataset", "cams": ("image", "wrist_image"), "robot": "franka",
                      "control": EEF, "hz": 20, "grip_dim": 6},
    "austin_sailor": {"repo": "lerobot/austin_sailor_dataset", "cams": ("image", "wrist_image"), "robot": "franka",
                      "control": EEF, "hz": 20, "grip_dim": 6},
    "stanford_hydra": {"repo": "lerobot/stanford_hydra_dataset", "cams": ("image", "wrist_image"), "robot": "franka",
                       "control": EEF, "hz": 10, "grip_dim": 6},
    "berkeley_fanuc": {"repo": "lerobot/berkeley_fanuc_manipulation", "cams": ("image", "wrist_image"),
                       "robot": "fanuc", "control": EEF, "hz": 10, "grip_dim": 6},
    "berkeley_ur5": {"repo": "lerobot/berkeley_autolab_ur5", "cams": ("image", "hand_image"), "robot": "ur5",
                     "control": EEF, "hz": 5, "grip_dim": 6},
    "iamlab_cmu": {"repo": "lerobot/iamlab_cmu_pickup_insert", "cams": ("image", "wrist_image"), "robot": "franka",
                   "control": "absolute end effector", "hz": 20, "grip_dim": 6},
    "droid": {"repo": "lerobot/droid_1.0.1", "cams": ("exterior_1_left", "wrist_left"), "robot": "franka",
              "control": "joint position", "hz": 15, "grip_dim": 7, "file_stride": DROID_FILE_STRIDE},
}


def clean_task(t):
    """LeRobot OXE ports carry RLDS artifacts in the task text: `tf.Tensor(b"...", shape=(), dtype=string)` wrappers
    and literal backslash-n sequences (seen in utaustin_mutex)."""
    t = t.strip()
    if t.startswith('tf.Tensor(b"') or t.startswith("tf.Tensor(b'"):
        t = t[len('tf.Tensor(b"'):]
        t = t[: t.rfind(", shape=")].rstrip("\"'")
    return " ".join(t.replace("\\n", " ").replace("\n", " ").split())


def condition(cfg):
    return f"robot: {cfg['robot']}; control: {cfg['control']}; {cfg['hz']} Hz"


def quantiles(x):
    return np.quantile(x, 0.01, 0), np.quantile(x, 0.99, 0)


def normalize(x, q):
    lo, hi = q
    return np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -1, 1).astype(np.float32)


# ---------------------------------------------------------------- LIBERO (re-rendered from the recorded states)

def libero_files(cfg):
    return sorted(f for su in cfg["suites"] for f in glob.glob(str(S / "raw" / su / "*.hdf5")))


def demo_names(h, max_eps):
    return sorted(h["data"], key=lambda s: int(s.split("_")[1]))[: max_eps or None]


def libero_columns(cfg, max_eps):
    """(state, action, task) per demo from the HDF5s, for the stats pass. The recorded proprio is one step later than
    the re-rendered one, which does not matter for the quantiles."""
    out = []
    for f in libero_files(cfg):
        with h5py.File(f, "r") as h:
            task = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
            for name in demo_names(h, max_eps):
                d = h["data"][name]
                st = np.concatenate([d["obs/ee_pos"][:], d["obs/ee_ori"][:], d["obs/gripper_states"][:]], 1)
                out.append((st.astype(np.float32), d["actions"][:].astype(np.float32), task))
    return out


def libero_episodes(cfg, max_eps):
    from libero.libero import get_libero_path

    from ecvla.libero_env import make_env

    for f in libero_files(cfg):
        suite = Path(f).parent.name
        bddl = os.path.join(get_libero_path("bddl_files"), suite, Path(f).stem.removesuffix("_demo") + ".bddl")
        env = make_env(bddl, image_size=IMG_SIZE, cameras=True)
        with h5py.File(f, "r") as h:
            task = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
            for name in demo_names(h, max_eps):
                d = h["data"][name]
                st, main, wrist = [], [], []
                for s in d["states"][:]:
                    obs = env.set_init_state(s)
                    st.append(libero_state(obs))
                    a, w = libero_images(obs)
                    main.append(a)
                    wrist.append(w)
                yield {"task": task, "state": np.stack(st), "action": d["actions"][:].astype(np.float32),
                       IMG[0]: np.stack(main), IMG[1]: np.stack(wrist)}
        env.close()


# ---------------------------------------------------------------- LeRobot hub ports

def lerobot_selection(cfg, max_eps):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(cfg["repo"])
    n = meta.total_episodes
    if "file_stride" in cfg:  # whole video files, spread over the dataset
        fidx = np.asarray(meta.episodes[f"videos/observation.images.{cfg['cams'][0]}/file_index"])
        chosen = set(sorted(set(fidx.tolist()))[:: cfg["file_stride"]])
        eps = [i for i in range(n) if fidx[i] in chosen]
    else:
        eps = list(range(n))
    return meta, eps[: max_eps or None]


def task_text(meta, task_index):
    """meta.tasks: DataFrame indexed by the task string, with a `task_index` column."""
    return meta.tasks.index[meta.tasks["task_index"] == int(task_index)][0]


def lerobot_columns(cfg, max_eps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta, eps = lerobot_selection(cfg, max_eps)
    ds = LeRobotDataset(cfg["repo"], episodes=eps, download_videos=False)
    hf = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(hf["episode_index"])
    st, ac, ti = np.stack(hf["observation.state"]), np.stack(hf["action"]), np.asarray(hf["task_index"])
    out = []
    for e in eps:
        m = ep == e
        if m.any():
            out.append((st[m].astype(np.float32), ac[m].astype(np.float32), task_text(meta, ti[m][0])))
    return out


def lerobot_episodes(cfg, max_eps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames

    meta, eps = lerobot_selection(cfg, max_eps)
    ds = LeRobotDataset(cfg["repo"], episodes=eps, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    ep_col = np.asarray(hf["episode_index"])
    for e in eps:
        idx = np.nonzero(ep_col == e)[0]
        if not len(idx):
            continue
        rows = hf[idx.tolist()]
        ts = np.asarray(rows["timestamp"], float).tolist()
        epm = ds.meta.episodes[int(e)]
        frames = []
        try:
            for cam in cfg["cams"]:
                key = f"observation.images.{cam}"
                path = ds.root / ds.meta.get_video_file_path(int(e), key)
                off = epm[f"videos/{key}/from_timestamp"]
                v = decode_video_frames(path, [off + t for t in ts], ds.tolerance_s, "pyav", return_uint8=True)
                frames.append(np.stack([resize_pad(x.permute(1, 2, 0).numpy()) for x in v]))
        except FrameTimestampError:  # the port's video does not cover the episode (berkeley_ur5 ep ~300); LeRobot
            SKIPPED.setdefault(cfg["repo"], []).append(int(e))  # advises ignoring such items
            print(f"{cfg['repo']}: episode {e} skipped (video does not cover its timestamps)", flush=True)
            continue
        yield {"task": task_text(meta, rows["task_index"][0]), "state": np.asarray(rows["observation.state"], np.float32),
               "action": np.asarray(rows["action"], np.float32), IMG[0]: frames[0], IMG[1]: frames[1]}


# ---------------------------------------------------------------- stats, signs, writing

def source_stats(cfg, cols, tok):
    st = np.concatenate([c[0] for c in cols])
    ac = np.concatenate([c[1] for c in cols])
    qs, qa = quantiles(st), quantiles(ac)
    g = cfg["grip_dim"]
    first = np.array([normalize(c[1][:1], qa)[0, g] for c in cols])  # the gripper starts open
    cond = condition(cfg)
    texts = [PROMPT.format(condition=cond, task=clean_task(c[2])) for c in cols]
    vals, counts = np.unique(np.round(ac[:, g], 2), return_counts=True)
    return {
        "condition": cond, "episodes": len(cols), "frames": int(len(ac)), "state_dim": int(st.shape[1]),
        "action_dim": int(ac.shape[1]), "state_q01": qs[0].tolist(), "state_q99": qs[1].tolist(),
        "action_q01": qa[0].tolist(), "action_q99": qa[1].tolist(), "grip_dim": g,
        "grip_sign": -1 if first.mean() > 0 else 1,
        "grip_first_frame_normalized_mean": float(first.mean()),
        "grip_first_frame_share_positive": float((first > 0).mean()),
        "grip_action_top_values": [[float(v), int(c)] for c, v in sorted(zip(counts, vals), reverse=True)[:5]],
        "empty_tasks": int(sum(not clean_task(c[2]) for c in cols)),
        "max_task_tokens": max(len(tok(t)["input_ids"]) for t in texts), "example_prompt": texts[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, choices=list(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0, help="per source (per LIBERO file); 0 = all")
    ap.add_argument("--check", action="store_true", help="stats, gripper signs and prompt lengths only")
    ap.add_argument("--stats", help="mixture_check.json of a full --check: use its per-source stats and the mixture's "
                                    "dims (building sources as separate parts for scripts/exp0005_merge_mixture.py)")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"fps": FPS, "img_size": IMG_SIZE, "prompt_format": PROMPT, "sources": {}}
    if a.stats:
        ref = json.loads(Path(a.stats).read_text())
        assert ref["prompt_format"] == PROMPT and ref["img_size"] == IMG_SIZE, "check file from another builder version"
        stats.update(state_dim=ref["state_dim"], action_dim=ref["action_dim"], stats_from=a.stats,
                     sources={n: ref["sources"][n] for n in a.sources})
    for name in [] if a.stats else a.sources:
        cfg, t0 = SOURCES[name], time.time()
        cols = (libero_columns if cfg.get("loader") == "libero" else lerobot_columns)(cfg, a.max_episodes)
        s = stats["sources"][name] = source_stats(cfg, cols, tok)
        print(f"{name}: {s['episodes']} eps, {s['frames']} frames, state {s['state_dim']}, action {s['action_dim']}, "
              f"grip sign {s['grip_sign']} (first-frame mean {s['grip_first_frame_normalized_mean']:+.2f}, "
              f"share>0 {s['grip_first_frame_share_positive']:.2f}), max tokens {s['max_task_tokens']}, "
              f"empty tasks {s['empty_tasks']} ({time.time() - t0:.0f} s)", flush=True)
    if not a.stats:
        stats["state_dim"] = max(s["state_dim"] for s in stats["sources"].values())
        stats["action_dim"] = max(s["action_dim"] for s in stats["sources"].values())
    (out / ("mixture_check.json" if a.check else "mixture_stats.json")).write_text(json.dumps(stats, indent=1))
    if a.check:
        print("check written", out / "mixture_check.json")
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sd, ad = stats["state_dim"], stats["action_dim"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (sd,), "names": None},
        "action": {"dtype": "float32", "shape": (ad,), "names": None},
        **{k: {"dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channel"]}
           for k in IMG},
    }
    ds = LeRobotDataset.create(repo_id=f"local/{out.name}", root=out / "data", fps=FPS, features=features,
                               robot_type="mixed", use_videos=True)
    for name in a.sources:
        cfg, s = SOURCES[name], stats["sources"][name]
        qs = (np.array(s["state_q01"]), np.array(s["state_q99"]))
        qa = (np.array(s["action_q01"]), np.array(s["action_q99"]))
        gen = libero_episodes if cfg.get("loader") == "libero" else lerobot_episodes
        n, t0 = 0, time.time()
        for e in gen(cfg, a.max_episodes):
            task = clean_task(e["task"])
            if not task:
                continue
            st = np.zeros((len(e["state"]), sd), np.float32)
            st[:, : e["state"].shape[1]] = normalize(e["state"], qs)
            act = np.zeros((len(e["action"]), ad), np.float32)
            act[:, : e["action"].shape[1]] = normalize(e["action"], qa)
            act[:, cfg["grip_dim"]] *= s["grip_sign"]
            text = PROMPT.format(condition=s["condition"], task=task)
            for t in range(len(act)):
                ds.add_frame({"observation.state": st[t], "action": act[t], IMG[0]: e[IMG[0]][t],
                              IMG[1]: e[IMG[1]][t], "task": text})
            ds.save_episode()
            n += 1
            if n % 50 == 0:
                print(f"{name}: {n} episodes written ({time.time() - t0:.0f} s)", flush=True)
        s["episodes_written"] = n
        s["episodes_skipped_video"] = SKIPPED.get(cfg.get("repo"), [])
        print(f"{name}: done, {n} episodes ({time.time() - t0:.0f} s)", flush=True)
    ds.finalize()
    (out / "mixture_stats.json").write_text(json.dumps(stats, indent=1))
    print("done", out)


if __name__ == "__main__":
    main()
