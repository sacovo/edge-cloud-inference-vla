# Research Journal

Newest entries first. Each entry covers what was tried, the result, and what failed and why.

## 2026-10-28: A0 Phase A dataset built and quality-checked

- Dataset: `/scratch/sandro.covo/ecvla/phaseA/<suite>/<task>.h5`. 1,500 demos and 200,485 control steps (Spatial 62k, Object 75k, Goal 64k), 7.6 GB. Contents: 500 Hz torque/F/T/joint/gripper, 20 Hz object poses, 224 px JPEG agent-view + wrist renders, actions, sim states. The full build (30 × 4 CPUs) took ≤1 h 15 min per task. Summary: `experiments/EXP-0001-latent-taskid-vs-state/results/a0_phaseA_dataset_summary.json`.
- Open-loop replay success: 92–93% per suite. Not needed, since training uses the hybrid (state-reset) replay; listed only as a sanity check.
- Quality sidecar `/scratch/sandro.covo/ecvla/phaseA_quality/` (`src/ecvla/quality.py`):
  - **Demos whose recorded states ever reach the success predicate:** Spatial 500/500, Object 500/500, Goal 480/500. All 20 misses are in `put_the_wine_bottle_on_the_rack`: those demos end with the bottle still in the gripper. Borderline cases flip by machine (30/50 on the cluster vs 33/50 locally), so this task's predicate is fragile. Treat it with care in evaluation, or exclude it from Goal analyses.
  - **Demos with ≥1 physically unstable replay step** (sign-invariant qpos error > 0.05, e.g. objects jumping in tight contacts): Spatial 61, Object 42, Goal 127 (of 500 each). They are flagged per step, so training can mask torque/F/T on those steps.
  - Earlier max-error outliers (1.82) were real instability, not quaternion sign flips.
- Lessons on test runs:
  - The quality array OOM'd at 4 GB on 6 Object tasks after a Goal-task test passed. Memory grows per demo with LIBERO hard resets (peak 4.5 GB), so test runs should use the **heaviest** suite.
  - Rerun with 12 GB; skip-existing made the rerun cover only the failed tasks.
- Next: GPU work (backbone latency on H200, then the FAST fine-tune), and the fast-policy training code for A1 (oracle pilot).

## 2026-10-28: A0 on the cluster: env, download, dataset build running

- The author asked for no expensive work on the login node (I had run `uv sync` and a download there). Setup and download are now jobs (`scripts/slurm/setup_env.sbatch`, `a0_download.sbatch`), with a shared job env in `scripts/slurm/env.sh`.
- First debug job failed: the login-node venv pointed at a system Python the compute nodes lack, and `uv run --frozen` without `--extra libero` dropped robosuite. Fixed with a uv-managed Python and the explicit extra.
- egl-probe (hf-libero → robomimic) needs CMake <4 to build. Provided via `tool.uv.extra-build-dependencies`.
- CPU nodes render with Mesa EGL (llvmpipe). Tests pass on a compute node, including image alignment.
- Download: 30 task files (Spatial/Object/Goal), 19 GB on `/scratch/sandro.covo/ecvla/raw`.
- Array-script test (task 0, 2 demos, cpu-debug): 276 steps in 119 s (0.43 s/step at 224 px), 34 KB/step, 2/2 open-loop success. Full array 119313 (30 tasks × 4 CPUs) submitted.
- Added `src/ecvla/latency.py` (latent delivery schedule: period, delay, jitter, drops, age) with tests.
- Decisions: LeRobot PyTorch `lerobot/pi05_base`; onboard target Jetson Orin NX/Nano. Tight for 3B, so the slow loop may run at ≤1 Hz.

## 2026-10-28: A0 started: LIBERO replay works

- Compute: FHNW HPC (`calculon`, Slurm; 2 nodes × 4 H200, plus RTX A4500/3080/2080 nodes). Use `/scratch/sandro.covo/hf` (NVMe) for data. DROID 1.0.1 and the FAST tokenizer are already cached there, which is useful for the backbone stage. No conda; the project uses uv. Locally: an AMD ROCm GPU, with MuJoCo rendering through Mesa EGL.
- `HuggingFaceVLA/libero` (LeRobot format) **cannot** be used for replay: it is 10 fps, with an 8-D state and no simulator states. The original LIBERO HDF5 files (`yifengzhu-hf/LIBERO-datasets`) have per-step 92-D MuJoCo states, actions and 128×128 images.
- Stack: hf-libero 0.1.4 + robosuite 1.4.0 + mujoco 3.8.1 (same as smolvla-rl). The project uses its own LIBERO config via `LIBERO_CONFIG_PATH`, because the global `~/.libero` points at another project.
- Replay check (`scripts/a0_replay_check.py`, LIBERO-Spatial task 0, 5 demos):
  - **hybrid** (reset to each recorded state, apply the recorded action, log all 25 physics substeps = 500 Hz): median per-step qpos error ~1e-3. The max of 0.028 is in object free-joint quaternions at contact events (grasp, place), i.e. a few degrees. Robot joints are accurate, and nothing accumulates because each step resets.
  - **open loop** from the initial state: 5/5 succeed, with max drift 0.13.
  - Gripper F/T sensors (`gripper0_force_ee/torque_ee`) and joint actuator torques are available at 500 Hz. F/T is never zero (gripper weight), so it needs bias compensation.
  - Result file: `experiments/EXP-0001-latent-taskid-vs-state/results/a0_replay_check_spatial_task0.json`.
- Next in A0: replay the full Spatial/Object/Goal suites into a dataset (500 Hz substeps + 20 Hz object poses + images); the z delay wrapper and its tests; and backbone latency.

## 2026-10-28: Two-stage latent (FAST fine-tune, then freeze); listening-robot idea parked

- Decision (author): train the backbone. A frozen backbone performs worse, both in the author's experiments and per π*0.6 / RECAP (arXiv 2511.14759). Autoregressive FAST+ action-token fine-tuning helps a lot.
- Two-stage pipeline: fine-tune `pi05_base` on FAST tokens using diverse robot data, then freeze it and train the fast-policy grid on its fixed latent. One backbone run serves the whole grid.
- Primary latent `z-ft-div` excludes LIBERO from fine-tuning, consistent with avoiding the `pi05_libero` skew. Variants: `z-ft-lib` (adds LIBERO train demos) and `z-frozen` (raw, ablation). Readouts: R1 (prefix, one forward pass, primary) and R2 (first 8 decoded FAST-token states). New `slow-only` reference: the fine-tuned backbone used directly as a policy.
- Added `literature/bib/06-experiment-design.bib` (FAST, π*0.6, PaliGemma 2), generated from arXiv metadata; the hook verified it.
- Parked idea: an actively listening robot (slow conversation-context latent + fast backchannel model on audio cues). See `docs/ideas/active-listening-robot.md`.

## 2026-10-28: EXP-0001 starts on LIBERO (Phase A)

- Decision (author): start with an existing suite and existing demos, and build our own dataset only if the concept shows up.
- Phase A runs on LIBERO, replaying the demos to get multi-rate sensors, torque, F/T and ground-truth state. LIBERO-Goal (fixed layout, so task ID should suffice) acts as a control for Spatial/Object (where layout or object varies, so scene state is needed).
- Pre-registered Phase A criteria: A-test (does the suite need state?), A-z (does z recover ≥50% of the oracle gap?), and A-control (does z help less on Goal?). Phase B (bootstrapped event/dynamic tasks) starts only if A-z and A-control pass.
- Risk noted: `pi05_base` pretraining may include LIBERO. Check this in A0.
- Next: A0 infrastructure.

## 2026-10-28: EXP-0001 decisions (slow model, demos, onboard profile)

- Slow model: `pi05_base` only, not `pi05_libero`, which was fine-tuned on LIBERO and would skew z. Added PaliGemma 2 28B as the generic ~27B remote model.
- The author noted that π0.5 (~3B) fits on a Jetson but is too slow to output actions for control. That leads to a second deployment profile: both loops on the Jetson, with the 3B model as a slow loop emitting only z and a tiny fast policy on top. This is the same research question with a different latency model (fixed compute delay instead of network jitter), so it went into EXP-0001 as profile P-onboard rather than a new experiment.
- Demos: teleop a few seed episodes, train ACT or SmolVLA, keep successful rollouts, retrain, and add manual episodes where needed. The author's RECAP-on-SmolVLA work is an option for improving the collector. Watch for success-filter bias toward easy states and too few recoveries.

## 2026-10-27: EXP-0001 designed (task-ID vs state crossover)

- Scope decision: manipulators are the primary platform, because training data and sim environments are plentiful. Drones were one example and are now optional (E4).
- Author's hypothesis: a large fast policy uses the MLLM latent only as a task ID. Below some size the latent has to carry scene state, the task-ID substitute fails, and staleness starts to matter.
- Author's key ablation: a fully blind fast policy (proprio + torque + wrist F/T). All scene state must then come through z.
- Added a delayed-oracle condition, so that "z lacks state" can be told apart from "state too old".
- Design fixed in advance in `experiments/EXP-0001-latent-taskid-vs-state/README.md`: hypotheses H0–H3, pass/fail criteria, and stages S0–S3. S1 is an oracle-only pilot with no VLM, which checks that the tasks need scene state before any VLM compute is spent.
- Next: S0. Resolve the open decisions (which VLM, how to generate demos for the event/dynamic tasks, LIBERO defaults) and build the delay/multi-rate env wrapper.

## 2026-10-27: Rapid scoping review completed

- Ran the review per `literature/PROTOCOL.md`: 5 themes, each searched by its own agent. The result is 147 deduplicated references, with every arXiv ID checked against the arXiv API.
- Closest prior work: AsyncVLA (arXiv 2602.13476), CloudEdgeVLA (arXiv 2608.00569), Figure Helix. Key gaps:
  - no fast-policy size sweep under a VLM latent
  - no explicit latent-age input under jitter
  - no fusion of IMU, torque, camera and a remote latent
  - no drone platform
  - no slow model around 27B
- Main threat to the experiments: the latent may encode only which task is being done (OpenHelix). The size sweep needs no-latent and task-ID-only baselines.
- Tooling: added a hook that verifies the bibliography. It caught that a wrong title with a reused arXiv ID was silently dropped as a duplicate during the merge, so `verify_bib.py` now checks the per-theme source files directly.
- Next: E0 from `literature/SYNTHESIS.md` §5. Pick the simulator and the 27B VLM, then precompute embeddings for LIBERO.
