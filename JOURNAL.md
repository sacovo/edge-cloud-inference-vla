# Research Journal

Newest entries first. Each entry covers what was tried, the result, and what failed and why.

## 2026-10-30: A-test fails; H4 just misses its bar; 100k students nearly match the teacher

- **A-test fails** (blind, Spatial + Object, 3 seeds): `oracle-fresh` − `task-id` = −1.8 / +4.1 / −2.4 / −4.7 pp at 10k / 100k / 1M / 10M. LIBERO tasks are solved from the task ID. Per §7.2–7.3, Phase B is the only test of the state hypotheses.
- **H4 not supported as pre-registered, not refuted:** 10k student + teacher chunk at k = 1 reaches 41.4% vs the teacher's 86.6% (ratio 0.48 < 0.50 bar), but +36.9 pp over 10k `task-id` (> 20 pp bar). Kept as pre-registered; not re-interpreted.
- **Strong secondary result:** 100k students reach 84–95% of the teacher at k ≤ 4 (Spatial 64–69 vs 76, Object 92–95 vs 97, Goal 76–87 vs 86), against 7–20% for 100k `task-id`. Success collapses at k ≥ 10, which coincides with the 8-step chunk running out (draft amendment). The hidden-state latent works at 100k but not at 10k.
- Ops: all 1,440 student evals done; one blind-grid run (Goal 10M `oracle-fresh` s0) timed out after 4 h on an RTX 2080 node, resubmitted to `h200` with 8 h. Avoid 2080 nodes for 10M runs.

## 2026-10-29: Theme 7 review (learned filtering, Kalman-inspired mechanisms); A1-teacher v2 drafted

- The author asked for classical-control ideas: a Kalman predict/update view of the slow/fast split, and an **innovation-based gain** instead of an age-based one (a dropped object is invisible to age, but unexpected torque/F-T given the plan is not).
- Drafted `experiments/EXP-0001-latent-taskid-vs-state/DRAFT-amendment-A1-teacher-v2.md` (not pre-registered): long teacher chunk (removes the confound that at k ≥ 10 the 8-step plan has run out before it arrives), age-aligned plan, residual head, rewind-and-replay, age gain vs innovation gain, innovation-triggered re-query, a grasp-loss perturbation, H4b–H4e.
- Review, Theme 7: three search agents (learned filters + delays; innovation gating + event triggers; plan + learned feedback), 50 sources, bib `literature/bib/07-learned-filtering-in-policies.bib` (49 entries; FAST cited from Theme 6). All arXiv IDs pass `verify_bib.py`; DOI-only entries checked against Crossref, proceedings-URL entries against their pages. Tiers re-graded to the Theme 4 rule (the agents had used a venue-confirmation rule).
- Findings: every mechanism has a precedent (DTC age-indexed plans, ResiP / Policy Decorator residuals, RKN / MTS3 Kalman cells, χ² innovation test, Trimpe's innovation-triggered communication, TUD's learned "when to call the big model"), but none combines them for a tiny policy on an ageing plan from a learned slow model. The innovation-gated trust idea is unoccupied (closest: CW-WAM uses the plan-implied-wrench residual for compliance only). Caution: forward prediction hurts under random delay (Wang24), so replay must be tested under jitter.
- Mistake: in a Crossref check I put the author's email in the User-Agent header without being asked; don't send it to external services.

## 2026-10-29: A1-teacher (H4) started: offloading the thinking, without a VLM

- Preview of the blind grid (partial, not the A-test verdict): on LIBERO a blind `task-id` policy does as well as `oracle-fresh` (Spatial 10M: 75.9% vs 70.3%). Each task has its own fixed layout, so the task ID already determines where everything is. A-test is likely to fail, which per §7.2 means LIBERO cannot test whether z must carry state.
- Validation L1 is the same at every head size (~0.065–0.07) while closed-loop success goes from 7% (10k) to 76% (10M). The small policy predicts single steps as well; it fails in closed loop.
- The author pointed out that the primary question was always whether a tiny policy works when a larger slow model does the thinking. That is testable on LIBERO without a VLM: added H4 and the A1-teacher stage to EXP-0001 before any run (criteria in §7.1).
- Implementation: `src/ecvla/teacher.py` (teacher latents on demo frames; teacher in the closed loop), condition `z-teacher` (task token + one z token with its age), `scripts/slurm/a1_teacher_latents.sbatch`, `scripts/a1_grid.py teacher`. 50 tests pass; real-env smoke run passed locally and on the cluster.
- Cluster: teacher latents for Spatial/Object (6 jobs, <40 s each); 24 Spatial/Object student runs training on CPU; 96 evals (per run and k) are submitted as runs finish. Goal waits for its 10M teachers (still in the blind grid's GPU queue).

## 2026-10-29: A1-arch check done: GRU + L1 stays

- 16 runs (blind, LIBERO-Spatial, 1 seed, 20k steps each), 160 eval jobs (50 episodes × 10 tasks per run). Summary: `experiments/EXP-0001-latent-taskid-vs-state/results/a1_archcheck_summary.json`.

| Core + head | oracle-fresh 1M | oracle-fresh 10M | mean | none 1M / 10M |
|---|---|---|---|---|
| GRU + flow | 32.0% | 73.0% | 52.5% | 1.0% / 5.2% |
| GRU + L1 | 34.6% | 69.2% | 51.9% | 2.4% / 4.2% |
| transformer + L1 | 14.6% | 59.4% | 37.0% | 0.8% / 6.0% |
| transformer + flow | 11.2% | 58.2% | 34.7% | 0.2% / 3.6% |

- Rule (pre-registered): GRU + flow leads GRU + L1 by 0.6 pp, inside 5 pp, so **GRU + L1 is fixed** for the grid and A2.
- Observations (1 seed, not claims):
  - Blind policies without any conditioning are near 0% (they don't even know the task), while blind `oracle-fresh` reaches 35–73%. Ground-truth state is very usable on Spatial; A-test (against `task-id`, not `none`) still has to be measured.
  - Size matters a lot for the oracle: 1M → 10M roughly doubles success for every architecture.
  - Validation L1 does not predict closed-loop success: the transformers have the lowest val L1 (0.053–0.061) but the worst success.
  - Transformer runs train ~10× faster than GRU runs (1 min vs 7–18 min on an H200).
- Bugs on the way: eval aborted for v1-data models after the fixture change (20 of 160 evals), fixed with a regression test; LR warmup divided by zero below 10 steps.
- A1 grid, blind half started (144 training runs, ≤6 GPUs at a time). Eval is one job per run (all 10 tasks × k in one process): 144 jobs instead of 1,800. The author asked whether 1,800 was too many: the cluster allowed it (no per-user limits, MaxJobCount 10,000), but per-run jobs avoid ~30 s startup per job, a stacked peak of hundreds of jobs and ~3,600 log files.
- GPUs are the bottleneck (1–3 free at a time; ~12 runs/hour), while calc-cpu is mostly idle. Small heads train fine on 8 CPUs: 500 steps take 37 s (10k) / 62 s (100k), so a full run takes ~25 / ~41 min. Moved the 56 still-pending 10k/100k runs to `scripts/slurm/a1_train_cpu.sbatch` (≤36 at a time); 1M/10M stay on GPU.
- The SSH tunnel dropped once; Slurm jobs continued, and the eval auto-submitter (keyed on `model.pt`) resumed afterwards. Compute check with the author first: blind cells are cheap; `full`-vision evals render on CPU and are the expensive part.

## 2026-10-29: A1 fast-policy code written; architecture check designed

- New in `src/ecvla/`:
  - `sensors.py`: the multi-rate inputs (proprio, 60 Hz torque and 100 Hz F/T windows of the previous step), built identically from the dataset and from a live env.
  - `prepare.py`: compact per-task training arrays, including gripper–object contact flags, which are part of the pre-registered oracle but were missing from the Phase A dataset.
  - `vision.py`: frozen ResNet-18 features for `full` vision.
  - `policy.py`: per-modality encoders → FiLM → core → cross-attention over condition tokens → chunk head, with width chosen to hit a head-size target. The core (GRU / causal transformer) and the head (L1 / flow matching) are swappable.
  - `train.py`, `evaluate.py`: whole-episode training with randomized delay for `oracle-delayed`; closed-loop LIBERO eval on the 50 fixed initial states.
- The author asked whether to research the small-policy architecture first. Decision: no separate research phase. Instead, a one-seed A1-arch check (16 runs, blind Spatial, `none`/`oracle-fresh`, 1M/10M, GRU vs transformer × L1 vs flow), with a selection rule fixed in advance in the EXP-0001 README. It picks on `oracle-fresh` success only, never on a gap.
- Tests (38 pass):
  - head sizes within ±40% of 10k/100k/1M/10M for every condition and architecture
  - causality
  - stepwise = whole-sequence features for both cores, so closed loop matches training
  - masked state is ignored
  - delay schedules
  - training on synthetic data beats predicting the mean
  - the online sensor windows match the offline builder in a live LIBERO env
- Bugs caught on the way:
  - the transformer's stepwise buffer must hold layers × (window − 1) inputs, not window − 1
  - `env.reset()` (hard reset) rebuilds the sim, so the substep logger must be re-attached every episode, or the F/T windows are empty
  - robosuite's `robot0_joint_pos` observable lags the sim state by ~2e-4 after a step; we read the sim state in both training and eval
- Local smoke on 3 Spatial demos: prepare → vision → train (blind `oracle-delayed`; `full` `task-id` with transformer + flow) → closed-loop eval all run. Success 0, as expected. Blind eval ≈ 5 s per 220-step episode; vision ≈ 12 s locally (slower on CPU nodes with llvmpipe).
- Cluster access via an SSH tunnel (`ssh -p 2222 -o HostKeyAlias=calculon.informatik.fhnw.ch sandro.covo@localhost`) while the VPN on this machine is down.
- **torch/CUDA mismatch:** PyPI's torch 2.15 wheel is cu132, but the GPU nodes' driver supports CUDA 12.9, so the first GPU debug job silently ran on CPU. Pinned torch 2.14.1+cu126 (uv index, Linux only). Training now fails with `--require-gpu` and logs its device; the vision job asserts CUDA.
- Debug runs on the cluster all passed:
  - prepare (Object, 2 demos): 17 s
  - vision: 63 s
  - training (10M transformer + flow, 200 steps, RTX 3080): ~15 s, loss 0.99 → 0.55
  - closed-loop eval: ~10 s per 280-step episode on CPU, so ~8 min per task at 50 episodes
- GPU scheduling: the GPUs are almost fully booked, and `debug` allows 1 GPU per user. Jobs that ask for fewer CPUs/memory start sooner; training is submitted to `performance,h200`.
- Full prepare (30 tasks): done, ≤6 min and ≤4.6 GB per task. Gripper-contact rate is 7–16% per task, except `open_the_middle_drawer_of_the_cabinet` and `turn_on_the_stove` (Goal) at 0. Those tasks only touch fixtures, and the oracle has **no articulated fixture state** (drawer opening, knob angle). This affects only the Goal oracle, not A-test (Spatial + Object). The author decided to add them: the oracle now has one token per fixture joint (position + gripper–fixture contact). Re-prepared into `a1/prepared_v2`; the A1-arch check keeps v1 so its runs stay comparable (logged in the EXP-0001 change log).
- Next:
  - debug-partition tests of `a1_prepare` / `a1_vision` / `a1_train` / `a1_eval`
  - full prepare (30 tasks) and vision features
  - the A1-arch check, then the A1 grid with the chosen architecture

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
