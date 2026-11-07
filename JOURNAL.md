# Research Journal

Newest entries first. Each entry covers what was tried, the result, and what failed and why.

## 2026-11-07: EXP-0005 slow-only = 0% — diagnosing model vs closed loop

- Tunnel down ~00:00–08:00 (no checks possible); jobs ran regardless.
- **Slow-only (030000, k = 4, hold, fp32): 0 / 500 successes** on the 10 finished tasks (Spatial t6–t9, Object t0–t5; 50 eps each), **0 decode failures**. Object t6–t9 (279651) still running, 0 successes after ~47 eps each. Gripper command means −0.23 … −0.63 on Object (the gripper does close), −0.98 / −1.00 on Spatial t8 / t6 (almost never closes). Arm action std ~0.2–0.35 normalized.
- Job problems: 279628 (Spatial t0) exited 134 (abort) right after loading the policy, no Python traceback; 279648 (packed Spatial t1–t5) decoded at ~90 s per chunk instead of 2.6–3 s in the other packed jobs on the same node and timed out at 6 h after 1–2 eps per task (cause unknown). Spatial t0–t5 have no result.
- At 5k the decoded chunks were already far from the demos (L1 0.46 normalized, dtype check). Training loss ended ~3.1 nats per FAST token.
- **Local checks (tunnel going down for hours; small files copied to `data/ecvla_scratch/exp0005/`):** LeRobot pre/post-processors are identity for state, images and actions (as intended; no double normalization); state and action 8-d on both sides, chunk 20. LIBERO training data in `mix_v2` (Spatial episodes 500/501 from the parquet) match what the slow-only loop computes **exactly**: normalized actions, normalized state re-rendered from the recorded states, prompt text (max |diff| 0.0). Merged dataset has 3,331,092 frames (not 3.9M as estimated from the per-source stats), so 30k × 16 is ~0.14 epochs. Image check (video frames vs local render) pending on the file-001 download; offline-check jobs 279766/279767/279768 queued (h200 ~11:44 / 12:15; debug A4500).
- **More local checks:** training images (mix_v2 video, Spatial episode 500, both cameras) vs the slow-only render of the same states: mean |diff| 2.7–3.2 (video compression), same orientation and brightness. FAST tokenize/detokenize code is consistent (DCT over time, (T, D) row-major, `min_token` −354, scale 10). Adapter checkpoint complete (252 LoRA tensors = 18 layers × 7 modules × A/B, plus `lm_head`), and LoRA did train: |B|_F 0.24–2.2 (init 0), |ΔW|_F (× alpha/r = 0.5) 0.14 (v) … 5.3 (gate) per matrix; relative size unknown without the base weights. Note `lora_alpha` 8 with r 16 at LR 2.5e-5 (LeRobot defaults; LoRA is usually run at a higher LR).
- **Calibration:** on LIBERO 1 s chunks (normalized, 1,974 frames from the parquet) the zero action has L1 0.397 and persistence 0.253. The 5k checkpoint's decoded chunks had L1 0.46 vs the demos (dtype check), i.e. **worse than the zero action** at 5k. The 30k offline check decides whether this persisted.
- So far no closed-loop pipeline bug found: inputs (state, images, prompt), normalization and the token path match training. The remaining candidate is the model itself (recipe: LoRA at full-FT LR, ~0.14 mixture epochs, ~1.2 LIBERO epochs), unless the offline check on 030000 is good.
- 10:00: Object t6–t9 (279651) finished: 0% each, so **slow-only 0 / 700** over 14 tasks. Offline checks 279766/279767 **aborted (exit 134) after ~3 min**, like slow-only 279628, all on calc-g-041 while other jobs of mine held GPUs there. Likely cause: robosuite's EGL context takes its device index from `CUDA_VISIBLE_DEVICES` (renumbered from 0 by Slurm) while EGL lists all of the node's GPUs, so a job on physical GPU k > 0 renders on another job's GPU. Fix in `vlm_env.sh` (a1a190c): `MUJOCO_EGL_DEVICE_ID=$SLURM_JOB_GPUS`, GPU variables logged at job start (to confirm the mapping). Possibly also behind 279648's 90 s decodes (not verified). Offline checks resubmitted: 279773 (030000), 279774 (005000); 279768 (A4500) still queued.
- **Correction (20:40):** the EGL-device explanation was wrong: the resubmitted checks 279773/279774 got `SLURM_JOB_GPUS=0` and still aborted (exit 134, ~3 min, no message; MaxRSS 24 GB of 48). All exit-134 jobs (279628, 279766/767, 279773/774) ran on **calc-g-041 and started after ~06:00 on 2026-11-07**; jobs that started there earlier finished (279607, 279649–279651), and 279648 (started 00:11 there) decoded ~30× slower than its siblings. The same offline check ran fine on calc-g-009. Most likely a fault on calc-g-041 (not verified; admins not contacted yet). EGL override reverted, GPU/host logging kept; exclude calc-g-041 for now.
- **Offline check 030000** (279768, calc-g-009 A4500, 139 frames: demo_0 of tasks 0/2/4/6/8 per suite, every 10th step; `$ECVLA_SCRATCH/exp0005/offline_check/ft_v2_b16_30k_030000_a4500_even.json`), L1 vs the demo's next 20 actions, normalized, pooled, per dim (pos xyz, rot xyz, grip): **model 0.28 / 0.37 / 0.39, 0.31 / 0.32 / 0.24, 0.34**; zero action 0.32 / 0.35 / 0.40, 0.22 / 0.26 / 0.27, 1.00; persistence (true current action repeated) 0.27 / 0.24 / 0.36, 0.26 / 0.25 / 0.15, 0.27. Gripper sign agreement 0.83. **On its own training frames the model's arm actions are no better than the zero action and worse than persistence in every dimension**: it barely uses the observation. The slow-only 0% is the model, not the closed loop (all closed-loop inputs were already verified to match training).
- **Author: LIBERO-only sanity run** before any recipe change. Same pipeline, data and LoRA settings, `mix_v2` with `LIBERO_SHARE=1.0` (sampler verified: the merged dataset's episode order matches `mixture_stats.json` at every source boundary, so the 30k run did sample 33% LIBERO), 10k steps × 16 ≈ 1.2 LIBERO epochs (the same LIBERO exposure as the 30k mixture run), schedule auto-scaled. Job 279807 → `runs/ft_v2_libonly_10k`; offline checks on 005000 / 010000 submitted automatically to a debug A4500 (same 5 tasks per suite as 279768). Reading: arm L1 well below the zero action and near or below persistence → the mixture run was diluted / too short; still ≈ zero → a training-setup problem to find first.
- Ops: calc-g-040 is held by a 2.5-day 4-GPU job, so h200 jobs land on calc-g-041 (other users' jobs run there normally).
- **Next:** `scripts/exp0005_offline_check.py` (decoded chunk vs demo per dimension on training frames, with zero-action and persistence baselines, gripper sign agreement) on 030000 and 005000 (jobs 279767 / 279766, h200). Offline good + closed loop 0% → closed-loop pipeline bug; offline poor → the model (undertrained: ~1.2 LIBERO epochs, or a training-data / normalization problem). Missing Spatial t0–t5 slow-only runs are on hold until this is known.

## 2026-11-06: EXP-0005 full FAST fine-tune submitted; slow-only k = 4 protocol and R1 extraction written

- **Full fine-tune:** 30k steps, batch 16 (author), job 279575 (h200, pending on resources at submission), run `$ECVLA_SCRATCH/exp0005/runs/ft_v2_b16_30k`, checkpoints every 5k.
- **LR schedule resolved:** LeRobot's pi0_fast cosine schedule (1k warmup, 2.5e-5 → 2.5e-6 over 30k) scales down for shorter runs, never up; any run > 30k needs `scheduler_decay_steps`.
- **Slow-only reference:** the old script replanned every 20 steps without delay; README §3 asks for the delayed k = 4 protocol. Added `--k/--hold-start/--dtype` reusing EXP-0003/0004's schedule code.
- **Stage 1:** `scripts/exp0005_latents.py` + `exp0005_latents.sbatch` (R1 at the last "Action: " token + mean-pooled prefix, frames re-rendered from recorded states).
- Debug tests of both on the 100-step smoke checkpoint, bf16, A4500 debug partition: jobs 279581 (latents; 279579 cancelled, asked for more memory than the free A4500 node had) / 279580 (slow-only). Start estimates: debug ~18:20–18:55, full fine-tune ~21:53 (calc-g-041).
- Access: VPN host did not resolve; tunnel used throughout.
- Debug results (smoke checkpoint, A4500): latents 279581 failed at CUDA init on calc-g-010 ("CUDA driver initialization failed"; nvidia-smi saw the GPU), so the node is excluded and the job resubmitted as 279586. Slow-only 279580 crashed in LeRobot's `detokenize_actions`: the 100-step model emits non-FAST tokens and the detokenizer asserts. A failed decode now counts as a dropped update (latest good chunk continues; no-op before the first) and is counted per episode (`decode_failed`); the dtype check does the same. Resubmitted as 279587 (slow-only) and 279588 (dtype). The debug partition allows 1 GPU per user, so these run one after another.
- 17:43 check: **full fine-tune 279575 running** on calc-g-041 (started ~17:37). Latents test 279586 passed on calc-g-009: 98 frames in 11 s (bf16, A4500), all finite, argmax at the R1 position is a FAST token for 100% of frames even on the 100-step model. Slow-only test 279587 runs end to end: 54 decodes in 220 steps, **all failed** (the 100-step model starts with "Subtask: …", apparently `pi05_base`'s subtask-prediction habit, not "Action:"), so the robot only did no-ops; 5.5 s per decode (A4500, runs to 256 tokens because no end marker comes). Expected for 100 steps; the 5k checkpoint will show whether the format is learned. The dtype test 279588 hit a stale line left over from the edit; fixed (41c5936), resubmitted as 279593.
- 18:13 check: fine-tune at ~2.4k steps, loss 15.3 (step 100) → 6.0 (500) → 4.5 (1k) → ~3.8 (2.4k), 0.685 s/step, 82 GB; only the known torchcodec/FFmpeg fallback messages in the log; ETA ~23:20. Dtype smoke test 279593 cancelled (debug GPUs busy until 22:31; the real check on `005000` runs the same code on h200 first).
- **bf16 vs fp32 on `005000`** (job 279607, h200, 66 frames: demo_0 of tasks 0 and 5 per suite, every 10th step; `$ECVLA_SCRATCH/exp0005/dtype_check/ft_v2_b16_30k_005000.json`): no failed decodes in either dtype (the format is learned by 5k). **R1 is unaffected** (cosine ≥ 0.996 for the last-token half, ≥ 0.9999 pooled; relative L2 3.1% / 1.1%). **Decoded chunks diverge** (identical on 11% of frames; L1 bf16 vs fp32 0.156 normalized, p95 0.48, vs FAST+ round-trip 0.017): greedy decoding branches after an early token flip. Quality against the demo actions is the same (L1 0.462 fp32 vs 0.466 bf16; both high at 5k). **bf16 is not faster** (0.91 vs 0.78 s per frame for R1 + one decode). So bf16 buys nothing; proposed: fp32 for slow-only and the latents (author to confirm).
- Author: fp32 for slow-only (and latents). Full slow-only (k = 4, hold, fp32, 50 eps × 10 tasks × 2 suites, one job per task) is submitted automatically when `checkpoints/030000` appears.
- **Full fine-tune 279575 done** (23:28, 5.9 h, no errors): final training loss ~3.1 nats per action token (15.3 at step 100, 4.5 at 1k, 3.55 at 5k, ~3.1–3.2 from 18k on; noisy per batch); checkpoint `$ECVLA_SCRATCH/exp0005/runs/ft_v2_b16_30k/checkpoints/030000`. Full slow-only submitted 23:31: jobs 279628–279647 (k = 4, hold, fp32, 50 eps per task, 10 tasks × 2 suites; out `slow_only/ft_v2_b16_30k_030000/k4_hold_fp32/`).
- 23:45: h200 queue would have run the 20 one-task slow-only jobs until Nov 9 (start estimates 02:50 → Mon 11:55). Kept 279628 (Spatial t0), replaced 279629–279647 with 4 packed jobs 279648–279651 (`exp0005_slow_only_pack.sbatch`: 5 tasks as parallel processes on one H200, same script and args, per-task logs `logs/ecvla-vlm-slowpack_<job>_<suite>_t<task>.out`).
- Fine-tune at 19:13: step 8k, loss ~3.43, on schedule.
- Author: compare bf16 vs fp32 on the first real checkpoint (`scripts/exp0005_dtype_check.py`: decoded chunks, action L1, R1 cosine). A local watcher submits it when `checkpoints/005000` appears. Cluster check every 30 min (session cron).

## 2026-11-06: Session handoff — EXP-0005 mixture built; full FAST fine-tune is next

- **Done:** mixture v2 (`$ECVLA_SCRATCH/exp0005/datasets/mix_v2`, 16 GB, 9,837 episodes, 9 sources, per-source quantiles and gripper signs in `mix_v2/mixture_stats.json`); weighted sampler (LIBERO 0.33, rest frames^0.43; `scripts/exp0005_train.py`, test passes); bf16 smoke tests (batch 16: 0.68 s/step, 82 GB; batch 32 + grad ckpt: 1.56 s/step, 45 GB).
- **Next (in order):**
  1. Decide the full-run length with the author (proposal: batch 16, 30k steps ≈ 6 h at 0.68 s/step ≈ 480k samples; LeRobot's pi0_fast schedule decays over 30k), then `sbatch -t 1-00:00:00 scripts/slurm/exp0005_train.sbatch $S/datasets/mix_v2 $S/runs/<name> --steps=30000 --save_freq=5000` (S = `$ECVLA_SCRATCH/exp0005`; extra args are not expanded in the job, pass absolute paths). The start checkpoint is built on the fly into `$S/init/pi0fast_from_pi05base_mix_v2` (needs mixture_stats.json).
  2. Slow-only check on LIBERO with the trained adapter: `scripts/slurm/exp0005_slow_only.sbatch --ckpt <run>/checkpoints/<N>/pretrained_model --mixture $S/datasets/mix_v2 --suite libero_spatial|libero_object --task-key <bddl name> --episodes 2 --out <json>`; then a proper slow-only success rate (precondition reference in §5).
  3. Stage 1: precompute R1 latents (hidden state at the first action-token position + mean-pooled prefix) on all LIBERO Spatial + Object demo frames, rendered as in the mixture (`scripts/exp0005_common.py`).
- **Known pitfalls:** VPN often drops; use the tunnel (`ssh -p 2222 -o HostKeyAlias=calculon.informatik.fhnw.ch sandro.covo@localhost`); background watchers must use the same host. VLM env = `vlm/` (separate uv project, torch 2.11 cu128, LeRobot @ f5c9bc0 as source archive); jobs that import `ecvla` need `PYTHONPATH=src`; do not use `sbatch --wrap` (dash), use `scripts/slurm/vlm_run.sbatch`. torchcodec cannot load FFmpeg on the nodes (harmless warnings; decode with pyav). `lerobot_train` with `--policy.path` needs the pre/post-processor files next to the weights; PEFT checkpoints load via `PeftModel.from_pretrained` on the start checkpoint (see `exp0005_slow_only.py`). OXE LeRobot ports: gripper 1 = open, some episodes' video does not cover their timestamps (skipped), task texts may carry `tf.Tensor(b"…")`.
- **Open / not verified:** whether LeRobot rescales its LR schedule to `--steps` (500-step debug run decayed fully); decode speed in bf16 for slow-only; `iamlab_cmu` gripper rule held for only 86% of episodes.

## 2026-11-04: EXP-0005 stage 0 — fine-tune pipeline works end to end; full mixture building

- **Debug FAST fine-tune learns** (500 steps, LoRA r 16 + LM head in full, 20+20 episodes): loss 14.1 → 4.9, below uniform over the FAST tokens (7.62) by step 100. Slow-only sim loop runs (2 episodes, 0 successes as expected; decode 1.8 s per 1 s chunk in fp32).
- **LIBERO data findings:** the HDF5 obs (proprio and images) are one step after the state the action was taken in; our sim renders LIBERO-Object darker than the recorded images and draws robosuite's gripper-site marker. Author: re-render LIBERO from the recorded states in our sim (training = eval images).
- **Mixture (author):** LIBERO + 7 OXE ports + a DROID file subset (~5k episodes, 16 of 312 video files), native formats and rates, rate in the prompt (common practice: RT-X, π0, CrossFormer, LeRobot; X-VLA normalizes to 4 s windows), video storage. mutex task texts carried `tf.Tensor(b"…")` artifacts (cleaned); every OXE port has gripper 1 = open in its first frames (flipped); 22% of the DROID subset has no instruction.
- SmolVLA (2506.01844) describes no rate handling; it filtered by embodiment, relabelled tasks with a VLM and fixed the camera order (§3.2).
- Full build running: 9 part jobs + merge (278831–278840). Next: weighted sampling (DROID ≈ all others combined), bf16, full LoRA fine-tune, slow-only, then stage 1 (R1 latents).
- Ops: `sbatch --wrap` lost `source` (dash) → `scripts/slurm/vlm_run.sbatch`; build jobs need `PYTHONPATH=src`; calc-g-002 has 95 GB, so memory requests matter there.

## 2026-11-03: EXP-0005 drafted and stage 0 started — pi05_base loads into LeRobot's pi0_fast

- **EXP-0005 (draft, criteria fixed):** how small an on-robot decoder can turn a frozen VLM latent into a usable plan. Author's decisions: `z-ft-lib` primary (OXE + LIBERO), LoRA first, closed loop at k = 4 (gating) and 10; decoder heads `raw` / `dct` chosen per size offline; H1 supported if a decoder ≤ 100k reaches ≥ 0.90 of the 1M decoder, refuted below 0.70; precondition 1M ≥ 50% of `slow-only`.
- **Fine-tuning recipe (author, after a quick literature brief, `cross_embodiment_brief.md`):** native action formats, per-dataset 1/99% quantile normalization, robot / control mode / rate in the prompt (π0.5 style). OXE action formats surveyed with sources (`oxe_action_formats.md`); LeRobot OXE ports have `robot_type` "unknown".
- **LeRobot code check:** `pi05` trains only with flow matching (no FAST loss); FAST training is `pi0_fast`. LeRobot needs torch < 2.12 → separate env `vlm/` (torch 2.11.0+cu128). Multi-dataset training is disabled and would merge stats → own per-dataset normalization step needed. Prompt: `Task: …, State: …;\nAction: `.
- **Load check passed** (job 125173): `pi05_base`'s VLM fills all 604 `pi0_fast` tensors (0 missing; 209 unused = action expert). FAST+ round-trips LIBERO 1 s chunks (32 tokens on average, L1 0.009). Open: whether `lm_head` already knows FAST tokens.
- Ops slips: a git dependency failed on the compute nodes (git-lfs required by the HPC git config, not installed) → LeRobot from the source archive; the HF token lacked gated-repo access (403, reported by transformers as "couldn't connect") → fixed by the author. Fallback if (a) had failed: the author's SmolVLA FAST+ code (`sacovo/smolva-rl`).

## 2026-11-02: EXP-0004 done — H1 supported: the k = 20 collapse was mostly the blind start

- **H1 supported as pre-registered** (100k, k = 20, Spatial + Object, 3 seeds): gap `replay` − `track` 17.9 pp without hold → **4.0 pp** with hold (seeds 6.0 / 0.2 / 5.7; 0.22 of the no-hold gap, bar ⅓).
- **Both arms jump with hold:** 100k `track` 22.4 → **61.1%** (0.89 of the teacher), `replay` 40.2 → 65.1%; 10k `track` 24.4 → 55.5%. At k = 10 the gap disappears (100k 74.2 vs 74.6%). So the k = 20 collapse in EXP-0002/0003 was mostly how episodes start, not plan staleness; a 1 s update period works if the robot waits for the first plan. EXP-0003's verdict stays as recorded (different protocol).
- **Residuals do not grow in closed loop** (100k k = 20 hold: 0.005 raw vs 0.025 teacher-forced). Without hold the residual spikes to 0.139 over the first 40 plan steps, then drops: the student fights a plan that no longer fits where its blind start took the robot. The student still adds nothing over the plan.
- 10k k = 20 keeps a 9.5 pp gap, driven by seed s1 (23.7; others 0.9 / 4.0).
- The no-hold rerun reproduces EXP-0003 exactly on Object, within 1.2 pp on Spatial (11 of 24 cells differ): the Spatial eval is not fully deterministic. Small against the effects; reported.
- This changes the earlier assessment ("stop pushing k = 20"): with a hold start, k = 20 is usable at 0.81–0.89 of the teacher.
- Author's decisions for EXP-0005 (2026-11-02): Open X-Embodiment for the FAST fine-tune, `z-ft-lib` primary, LoRA first, closed loop at k = 4 and 10.

## 2026-11-01: EXP-0004 (hold start) pre-registered and started

- Author's idea: before #4/#6, find out why the `track` student falls 17.9 pp short of `replay` at k = 20 in sim; hold the robot until the first plan arrives (both arms). Author's choices: first plan treated as fresh (ages 0..k−1), hold steps not counted, closed-loop residual logging added to the eval. Primary: 100k, k = 20; supported if the gap with hold is ≤ ⅓ of the gap without, refuted if ≥ ⅔.
- Code: `evaluate.py --hold-start` (schedule src −1 → 0; equivalent to a physical hold because the scene is static) and per-episode residual summaries for `track` students; test added (60 pass). Analysis script and manifests written before any run.
- Drift check (local, 100 episodes): a 20-step no-op hold after settling moves the EEF ≤ 2.0 mm / 0.23°, objects ≤ 2.5 mm.
- Debug 122623 passed; the no-hold rerun reproduces EXP-0003 exactly. In the 2 debug episodes, residuals without hold are large right after the first plan (0.11–0.12 raw in the first window) and small with hold (≤ 0.01); 2 episodes, not a result.
- Full runs 122857/122858/122859 (54 jobs) all started at once on c-005..007 (now free).
- Slips: the tunnel dropped mid-session (work continued locally; the VPN came back later). A background wait put a `python3` JSON read on the login node; stopped before it ran and replaced with a local read.

## 2026-10-31: EXP-0003 done — H1 inconclusive; the student falls short of the teacher replaying its own plan

- **H1 inconclusive as pre-registered** (100k, k = 20, Spatial + Object, 3 seeds): `track` 22.4% vs EXP-0002 11.6%, **+10.7 pp** (seeds +9.6 / +10.8 / +11.8; support needs ≥ 15, refutation < 5), **0.32** of the H40 teacher (needs ≥ 0.50). Consistent across seeds, just too small.
- **Reference arm `replay`** (teacher executes `plan[age]` of its own delayed plan): 76.8 / 76.3 / 70.8 / 40.2% at k = 1 / 4 / 10 / 20. Ratio k20/k1 = 0.52, just over the pre-registered 0.5, so by the rule the shortfall lies with the student; the plan alone also loses half. `track` − `replay` = −17.9 pp at k = 20, −8.1 at k = 10, within ±4 at k ≤ 4. `replay` at k = 1 beats the teacher's own closed loop (76.8 vs 68.9%, every seed), not explained.
- **Residual use (§6, post-hoc, held-out demos, teacher-forced):** the bound is never the limit (100k k = 20: 0.07% of steps at ≥ 95% of the bound; mean position residual 0.025 raw vs 0.5). The residual cuts L1 by 7% at k = 20 and never moves the gripper (~0.003 vs ±1). On demos student ≈ plan, so the closed-loop gap to `replay` is unexplained; candidates (not tested): closed-loop residual growth off the demo distribution, and the first k steps where the student acts alone vs `replay` using the teacher's fresh output.
- **10k student:** 5.9 → 66.3% at k = 10 (0.96 of the teacher), 21.8 → 73.1% at k = 1, within 4.5 pp of `replay` at k ≤ 10. Confounded with the window shrinking z 280 → 56-d (d 16 → 20); a 9k-param policy executes the 10M teacher's delayed plan about as well as the teacher does, up to k = 10. 100k at k ≤ 10: unchanged (−1.0 to +4.9 pp).
- Ops: replay, students and evals all completed on calc-cpu overnight (67 jobs, no failures; last eval 07:42, last replay 08:37). The residual check (job 122419, cpu-debug) never started (c-005..007 full, c-001..004 reserved; start estimate slipped 11:43 → 16:00); cancelled and run locally on a copy (`data/ecvla_scratch`: prepared_v2 55 MB, teacher_z 771 MB, students; ~20 min to copy over the tunnel, a few minutes to run).
- Results: `experiments/EXP-0003-tracking-structure/results/`, README §8. Next: queue #4 (late arrivals / rolling z forward) or #6 (latent compression); the author decides.

## 2026-10-30: Session handoff — EXP-0003 queued, nothing run yet

- **State at 15:30:** replay 122192 (6), students 122193 (12) and submitter 122194 all PENDING on calc-cpu (c-001..004 reserved for `aiub-prime`, c-006/007 full). Slurm start estimates: students ~18:00, replay ~06:00 tomorrow, submitter ~16:00 tomorrow. Time limits lowered to 3 h / 5 h / 10 h. Author: keep waiting, do not move to the GPU node.
- **Student evals only go out when the submitter runs**, or when someone runs `scripts/slurm/submit_ready_evals.sh experiments/EXP-0003-tracking-structure/manifests/student_eval.txt` on the login node (safe there; it shares the `.submitted` file with the submitter).
- **When everything is in** (240 replay files under `$ECVLA_SCRATCH/exp0002/runs/teachers/*/*/eval_replay`, 480 under `$ECVLA_SCRATCH/exp0003/runs/students/*/*/eval`): rsync the JSONs, then `a1_summarize.py <runs>/students --out students_track.json`, `a1_summarize.py <exp0002 runs>/teachers --eval-dir eval_replay --out replay.json`, `exp0003_summarize.py students_track.json replay.json --out exp0003_summary.json`; fill README §8, registry, QUEUE, journal.
- **Still owed for §6:** the fraction of steps at the residual bound (promised in §6; not computed yet; can be done on held-out demos like the EXP-0002 post-hoc check).

## 2026-10-30: EXP-0003 (tracking structure) pre-registered and started

- Author's choices: one combined arm (age-aligned 8-step plan window + bounded per-step residual), primary at 100k / k = 20 (supported if ≥ 50% of the H40 teacher and ≥ +15 pp over EXP-0002's 11.6%), act unbounded before the first plan. Reference arm `replay`: the teacher executes its own delayed plan (queue #3 note from the EXP-0002 post-hoc).
- Code: `PolicyConfig.track`/`track_bound`, `plan_window` (shared by training and closed loop), `--track`, `evaluate --replay` (results in `<teacher>/eval_replay`), `a1_summarize --eval-dir`. Tests: 60 pass, including zero-residual student == replay in closed loop. Analysis script `scripts/exp0003_summarize.py` written before any run.
- Debug: 10k `track` student d = 20, 9,036 params; val L1 0.123 after 200 steps (EXP-0002 debug student: 0.257). Replay at k = 20 succeeded on 2/2 episodes of Spatial task 0 (77 steps); the 200-step student 0/2.
- Observation (before any result, recorded in §6, not changed): the 0.5 raw bound is ~1 std on position but ~10 std on rotation, so it effectively bounds position only.
- Full runs: replay 122192 (6 teachers × 4 k), students 122193, submitter 122194 (48 evals). No GPU.
- Slip: read a run's info.json with `python3 -c` on the login node (trivial file read, but the rule is git/sbatch/squeue only; use `grep`/`cat` or fetch locally).

## 2026-10-30: EXP-0002 post-hoc — the plan goes stale gradually, success falls off a cliff

- The author suggested that k = 20 collapses because far plan steps no longer apply. Checked by teacher L1 per chunk step j on held-out demos (`scripts/exp0002_chunk_error.py`, job 122176, ~10 min on cpu-debug; j = 0 reproduces the logged val L1).
- H40, pooled: 0.066 (j = 0), 0.088 (j 4–7), 0.100 (j 10–19, used at k = 10), 0.122 (j 20–39, used at k = 20). Position and gripper degrade most; rotation is flat. Persistence (repeat a_t) is 0.371 at j 20–39, so old steps are still informative.
- Direction supports the author's view, magnitude does not explain the cliff on its own (+22% error vs 60.5% → 11.6% success). Open: closed-loop divergence (teacher-forced is a lower bound) or a grasp-precision threshold. Separating test: the teacher executing its own delayed plan at k = 20, without a student (candidate reference arm for queue #3; not added yet).

## 2026-10-30: EXP-0002 done — H1 supported: a 40-step plan rescues k = 10, not k = 20

- **H1 supported as pre-registered** (100k, k = 10, Spatial + Object, 3 seeds): H40 student 60.5% vs H8 38.7%, **+21.8 pp** (seeds +24.2 / +13.5 / +27.8), **0.88** of its own teacher. Seed s1 alone is under 15 pp; the criterion is on the mean.
- **The H40 teacher is weaker:** 68.9% vs 86.6% pooled (Spatial 56.7 vs 75.9, Object 81.1 vs 97.3); 9.19M vs 10.74M params at the 10M budget. Its 100k students match or beat it at k ≤ 4 (ratio 1.08–1.13).
- **k = 20 still collapses** (100k: 11.6%, ratio 0.17) although the plan covers age 39. The plan running out explains the k = 10 drop; the k = 20 one needs something else (a 1–2 s old plan is not usable open-loop; candidates are queue #3 tracking structure and #4 forward prediction).
- **The 10k student is worse with the 280-d z** (k = 1: 21.8% vs 41.4%; d 20 → 16) and gains nothing at k = 10. Supports doing queue #6 (latent compression) before asking 10k questions again.
- Ops: 6 teachers 15–19 min on h200; latents < 40 s; 12 students 16–38 min on CPU (submitted with `afterok` on the latent array; cross-cluster dependencies are impossible here, no federation); 48 evals ~30–45 min via the submitter (122009). No failures. Nodes calc-c-001..004 are reserved for another group (`aiub-prime`), so CPU jobs run on 005–007.
- Results: `experiments/EXP-0002-plan-horizon/results/` (summary + inputs), README §8.

## 2026-10-30: EXP-0002 started — debug runs pass; 6 H = 40 teachers queued

- **EXP-0001 leftover closed:** Goal 10M `oracle-fresh` s0 (job 277004, h200, 20 min) and its eval finished: 81.2% (s1/s2: 80.0/80.2%). `results/a1_grid_blind_summary.json` refreshed (one new row; the other 179 unchanged). Does not enter any EXP-0001 criterion.
- **EXP-0002 analysis fixed before any run:** `scripts/exp0002_summarize.py` (§5 pooling, ratio to own teacher, paired bootstrap over seeds). Fed EXP-0001's H8 data on both sides it reproduces H4 (10k k=1 41.4%, ratio 0.48, teacher 86.6%).
- **Debug runs (all pass, under `$ECVLA_SCRATCH/exp0002_debug`):** H40 teacher 200 steps on `debug` (2080 Ti, 87 s); its latents on `cpu-debug` (10 files, `z_chunk` 280-d, `z_hidden` 704-d); 10k student 200 steps (`latent_dim` 280, d = 16, 10,056 params); 2-episode eval at k = 10 with the H40 teacher in the loop (0/2, expected after 200 steps).
- **Widths (§6):** teacher H8 d = 768 (10.74M head params) vs H40 d = 704 (9.19M; the 40 × 7 output layer takes part of the 10M budget); 10k student H8 d = 20 vs H40 d = 16; 100k H8 d = 84 (H40: when trained).
- Full runs: 6 teachers (array 277058, `-p h200`, 8 h limit); cluster-side submitter 122009 watches `teacher_eval.txt` and `student_eval.txt`. Latents and students are submitted when the teachers finish.
- Slip: the first debug-teacher submission passed `--out \$ECVLA_SCRATCH/...` as an extra arg; extra args are not expanded inside the job's `eval`, so it would have written to a literal `$ECVLA_SCRATCH` dir. Cancelled within seconds (no dir created); pass absolute paths for extra args.

## 2026-10-30: Session handoff — EXP-0002 pre-registered; latent compression queued

- **EXP-0002 criteria approved by the author** before any run; status pre-registered. Next session starts its §7: debug runs (one H = 40 teacher for 200 steps on `debug`/`h200`, its latents, one student for 200 steps, a 2-episode eval), then 6 teachers, teacher evals, latents, 12 students, 48 evals. Manifests: `python3 scripts/exp0002_manifests.py` (paths under `$ECVLA_SCRATCH/exp0002`). Use the cluster-side submitter for evals.
- **Queued #6 (author):** latent compression — compress z to a few dimensions so it does not overwhelm a 10k student (the 768-d hidden latent failed at 10k in EXP-0001; EXP-0002's chunk latent is 280-d).
- **Still running on the cluster:** the last EXP-0001 blind-grid run (Goal 10M `oracle-fresh` s0, job 277004 on `h200`, 8 h limit) and the cluster-side submitter (job 121207, ends after submitting that run's eval). Not needed for any EXP-0001 criterion; when its eval lands, re-run `scripts/a1_summarize.py` on `runs/grid` and update `results/a1_grid_blind_summary.json`.
- Not done yet: full-vision eval timing (the 200-step vision model exists at `$ECVLA_SCRATCH/a1_test/vis_spatial_tid_1M`).

## 2026-10-30: Work restructured into single-question experiments; EXP-0001 closed; EXP-0002 designed

- The author asked to stagger the work: one question per experiment, each with its own result, combined at the end, instead of one large amendment. The 9-variant draft "A1-teacher v2" was retired (git 43cf56e) and split into `experiments/QUEUE.md` (plan horizon → tracking structure → late arrivals → innovation trust; VLM latents, state tasks and infrastructure as independent items).
- EXP-0001 closed after Phase A (results in its §10).
- Merged literature Theme 8 (JEPA / predictive latents, 19 sources, verified): a ~9K-parameter JEPA predictor runs at 100 Hz on a Jetson Orin NX (SkyJEPA), but nobody rolls a delayed remote latent forward; V-JEPA 2 features recover LIBERO object state better than SigLIP (vision towers only).
- EXP-0002 designed (`experiments/EXP-0002-plan-horizon/README.md`): teacher chunk 8 → 40 steps (covers ages up to 39 at k = 20), everything else as EXP-0001; primary criterion at 100k, k = 10. Chunk length is now a model setting (default 8; EXP-0001 teacher parameter count reproduced in a test). Not run yet.

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
