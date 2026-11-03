# EXP-0005: How small can an on-robot decoder be that reconstructs the plan from a VLM latent?

- **Status:** draft (2026-11-01); dataset, latent, recipe, k, decoder heads and all criteria decided by the author 2026-11-02; stage 0 (Open X-Embodiment subset choice) in progress
- **Platform:** LIBERO-Spatial and LIBERO-Object; slow model `pi05_base` (FAST fine-tuned, frozen); simulated delay
- **Related:** EXP-0001 §3 (slow model, backbone stage, readouts; designed, never run), EXP-0001–0003 (a 10k–100k student executes a 10M teacher's plan at k ≤ 4), `experiments/QUEUE.md` ("VLM latents").

The hypothesis, conditions and criteria below are fixed **before** any run. Changes after the first result go in the Change log with a reason.

## 1. Question

EXP-0001–0003 showed that a tiny policy can execute a big policy's delayed plan: at k = 4, 10k students reach 0.92 and 100k students 0.94 of the 10M teacher's success (EXP-0003 `track`; EXP-0001), while the same sizes fail without the plan (task ID: 4.5 / 7.7%; exact object state: 2.7 / 11.8%). But that teacher's plan is a linear readout of its hidden state, so with a teacher latent the decoding problem is trivial and the floor is set by the latent's width alone.

With a VLM latent the plan is not linearly attached: the latent has to be decoded into actions by something. The project's research question then becomes measurable: **how small can the on-robot network be that turns the slow model's latent into a usable plan?** That network sets a floor on the on-robot policy; a later experiment starts the student from it and adds the fast sensor inputs (out of scope here, §8).

## 2. Hypotheses

- **H1 (small decoder).** A decoder of ≤ 100k on-robot parameters reconstructs, from the frozen VLM latent, a plan that performs nearly as well in closed loop as the plan from a much larger (1M) decoder.
- **H0-size.** Plan decoding from a VLM latent needs more than 100k on-robot parameters (it is not a near-linear readout).
- **H0-latent (precondition).** The latent does not carry a usable plan at all, at any decoder size; then H1 is not evaluated (§5).

## 3. Design

### Slow side (fixed for the whole experiment)

- **Backbone** as EXP-0001 §3: `pi05_base` (LeRobot port), fine-tuned autoregressively on FAST+ action tokens with **LoRA** (author, 2026-11-02; full fine-tune only if LoRA fails the `slow-only` sanity check), then **frozen**. One backbone run serves the whole grid.
  - **Primary latent `z-ft-lib`** (author, 2026-11-02): fine-tuning data = an **Open X-Embodiment** mixture + the LIBERO training demos of the evaluated suites (never evaluation initial states). Changed from EXP-0001's `z-ft-div` primary because a backbone that never saw LIBERO actions could fail the precondition (§5) for reasons unrelated to decoder size. `z-ft-div` (same mixture, no LIBERO) is an optional secondary variant; `z-frozen` is not run.
- **Fine-tuning data recipe** (author, 2026-11-02; evidence in `cross_embodiment_brief.md`): as `pi05_base`'s own pretraining (2504.16054 §IV-C).
  - **Native action formats**, no conversion to LIBERO's: each dataset is normalized per action dimension to [−1, 1] by its own 1% / 99% quantiles and padded to the backbone's action dimension. Joint-space and absolute-pose datasets are therefore usable (candidate list and formats: `oxe_action_formats.md`).
  - **Condition in the text prompt**: robot, control mode and rate, e.g. `robot: franka; control: delta end effector; 20 Hz` (π0.5 marks the control mode in the prompt, §IV-C; RDT-1B feeds the rate, 2410.07864 App. D). The LeRobot OXE ports have `robot_type` "unknown", so the fields come from the OXE spreadsheet. The exact prompt format follows what the LeRobot `pi05` code expects (stage 0).
  - **Control rates (author, 2026-11-03):** each source at its native rate with the chunk counted in steps (a 20-step chunk is 2 s of a 10 Hz source), the rate in the prompt. Common practice: RT-X 2310.08864 §IV-C, π0 2410.24164 App. A-D, CrossFormer 2408.11812 §3.3, LeRobot's `resolve_delta_timestamps`; rate as input as RDT-1B 2410.07864 App. D and X-VLA 2510.10274 Table 10. Time-normalized chunks (X-VLA's 4 s anchors, §4.2.2; FAST+'s 1 s chunks, 2501.09747 §V-C) only if `slow-only` on LIBERO is weak.
  - **Aligned anyway:** gripper sign. **Excluded:** datasets whose documentation contradicts itself (`taco_play` gripper, `toto` action space) and ones without rotation (`jaco_play`, `language_table`).
  - **Weights:** LIBERO Spatial + Object training demos up-weighted (in-domain data beats OXE, 2412.14058 §II-E); the rest by size, n^0.43 (2410.24164 §V-A). The LIBERO weight is fixed in stage 0 from the mixture sizes, before the fine-tune.
- **Readout R1** (EXP-0001 §3): final-layer hidden state at the position that predicts the first action token, concatenated with the mean-pooled prefix. One forward pass, no decoding. Inputs: agent-view + wrist RGB and the instruction.
- **Slow-side projection.** A linear map from R1 to d_z ∈ {16, 64, 256}, trained jointly with each decoder but **run on the slow machine** (it is part of what is transmitted, and it sets the payload). Not counted in the on-robot size; its parameters and the payload bytes are reported.

### On-robot decoder (the variable)

- MLP: d_z → hidden → hidden → output, GELU, L1 loss on the normalized 40 × 7 plan (a 40-step chunk, as the EXP-0002/0003 teachers; same normalization and target construction as `train.py`). Input is z only: no proprio, no task token, no age (fresh plan, age 0; delay enters only in closed loop, below).
- **Two output heads** (author, 2026-11-02):
  - `raw`: the output layer predicts all 40 × 7 values.
  - `dct`: the output layer predicts the first n_c DCT-II coefficients per action dimension (7 × n_c outputs); a fixed, parameter-free inverse DCT rebuilds the 40 steps, and the L1 loss is on the rebuilt plan. Why: at small sizes the 280-wide output layer takes most of the budget (10k, d_z = 16: 29 hidden units, 86% of the parameters in the output layer; with n_c = 8: 69 hidden units, 40%). The basis is fixed and assumes only smooth plans, the premise of FAST's per-dimension DCT (`pertsch2025fast`, Sec. V-B, via Theme 7).
  - n_c is fixed **before training**: the smallest n_c whose truncated DCT reconstructs the training-split demo chunks to within 10% of the H40 teacher's held-out first-step L1 (EXP-0002: 0.066), i.e. reconstruction error ≤ 0.0066 in action units.
- **Sizes (on-robot parameters):** 3k, 10k, 30k, 100k, 300k, 1M (hidden width set per size, head and d_z). For each size, the head (`raw` / `dct`) and d_z are chosen by validation L1 on held-out demos (never on closed-loop success); only the chosen (head, d_z) goes to closed loop.
- **Data:** LIBERO-Spatial and -Object demos (`prepared_v2`), same train / held-out split as `train.py` (last 2 kept demos per task held out). z precomputed once for all demo frames.
- **Seeds:** 3 per (size, head, d_z), one backbone. Selection uses the mean validation L1 over seeds.

### Closed loop

- **`decoder-replay`:** the VLM (frozen) is queried every k = 4 steps from the current observation; its z arrives exactly k steps later; the decoder turns it into a 40-step plan, and the robot executes step `age` of the latest delivered plan (as EXP-0003's `replay`). Before the first plan: hold start as EXP-0004 (first plan treated as fresh), to keep start-up effects out.
- **k = 4 and 10** (author, 2026-11-02; k = 4 is the project's operating point, ~5 Hz slow updates at 20 Hz control; k = 10 is 2 Hz). EXP-0004 showed both are usable with a hold start for a teacher plan. Evaluation as EXP-0001–0004: 50 fixed initial states × 10 tasks per suite, eval seed 0.
- **References** (same protocol): `slow-only` (the fine-tuned backbone decodes its own FAST action chunk, executed the same way at k = 4) and the **1M decoder** (the reference in §5).

### Positive control (offline, cheap)

The same decoder grid on the EXP-0002 H40 teacher's hidden state (`z_hidden`, 704-d, already precomputed), offline L1 only. Its plan is a linear readout of that state, so small decoders should match large ones up to the d_z bottleneck. A failure here points at the pipeline, not the latent.

## 4. Measurements

- **Offline:** plan L1 on held-out demos per chunk step (as EXP-0002 post-hoc), per (latent, size, d_z, seed); the VLM's own decoded FAST chunk (`slow-only`) on the same frames as a reference.
- **Closed loop:** success per (suite, size, seed, k); pooled Spatial + Object per seed; ratio to the 1M decoder (mean / mean) with bootstrap 95% CI over seeds; per-seed values.
- **The size curve** (success vs on-robot parameters) is the main reported result.
- Payload (d_z × bytes) per update; slow-side projection parameters; R1 width.

## 5. Pre-registered criteria

Primary latent `z-ft-lib`, closed loop at **k = 4**, Spatial + Object pooled, 3 seeds. k = 10 is reported with the same statistics, not gating.

- **Precondition (H0-latent):** the 1M decoder reaches ≥ 50% of `slow-only`'s success. If not, the result is "the latent does not carry a usable plan" and H1 is not evaluated.

| | Supported if | Refuted if |
|---|---|---|
| **H1, primary** | some decoder ≤ 100k reaches ≥ 0.90 of the 1M decoder's success | no decoder ≤ 100k reaches ≥ 0.70 of it |

In between: inconclusive. **Why 100k and 0.90 / 0.70 (approved by the author 2026-11-02):** 100k is the size at which students matched their teacher from a teacher plan (EXP-0001/0002); the question is whether that survives a latent that must be decoded.

Reported, not gating: the smallest size reaching 0.90 (the floor); k = 10; `z-ft-div` if run; `slow-only` vs the 1M decoder; offline L1 vs closed-loop success across sizes; the positive control.

## 6. Threats to validity

- **One backbone run.** Seeds vary the decoder only; backbone variance is not measured.
- **The latent is in-domain** (`z-ft-lib` saw LIBERO training demos), so the floor is for a slow model tuned to the suite. `pi05_base` pretraining may also include LIBERO (EXP-0001 §5 risk); checked on its data card and logged.
- **The slow-side projection does real work.** Counting it as "remote" is a deployment choice (it runs where the backbone runs and cuts the payload). Reported separately so the on-robot floor can be read with and without it.
- **Readout R1 only.** A poor floor could be the readout; R2 (decoded-token states) is a later variant.
- **LIBERO needs no scene state** (EXP-0001 A-test), so the latent only has to carry the plan. The floor measured here is a floor for plan decoding, not for state extraction.
- **No robot sensors in the decoder** (by design; §8). The closed loop is the decoded plan executed open loop between updates, as `replay`.
- **Action-space mapping.** FAST+ tokens from diverse data must map consistently onto LIBERO's delta-pose controller (EXP-0001 §8 open item).

## 7. Plan and cost

0. **Infrastructure (prerequisite, not a test):** LeRobot `pi05_base` in the project env (check against the pinned torch 2.14.1+cu126); Open X-Embodiment mixture + LIBERO demos with FAST+ tokenisation; LoRA FAST fine-tune → `z-ft-lib` on H200; `slow-only` LIBERO success as a sanity check. Debug first: a few hundred steps on `debug` / `h200`, a 2-episode `slow-only` eval.
1. Precompute R1 for all Spatial + Object demo frames (one forward pass per frame, GPU).
2. Fix n_c from the demos; positive control and offline decoder grid (6 sizes × 2 heads × 3 d_z × 3 seeds × 2 suites; small MLPs on CPU).
3. Closed loop at k = 4 and 10: 6 sizes × 3 seeds × 2 suites × 2 k plus `slow-only` (GPU: the VLM runs in the loop every 4 steps). Cost estimated from the debug eval before submitting.
4. Analysis script written before any closed-loop run.

The backbone stage (0) is the expensive and uncertain part; its cost is estimated in the debug run and logged before the full fine-tune.

## 8. Out of scope (next experiment)

Starting the on-robot student from the smallest working decoder and adding the fast sensor inputs (proprio, torque, F/T, and later camera), compared against the same student trained from scratch. Its value can only show on tasks where the plan alone is not enough (EXP-0003/0004: on LIBERO the student mostly executes the plan), so that experiment also needs such a task.

## 9. Decisions

Decided by the author (2026-11-02): Open X-Embodiment mixture (+ LIBERO training demos) for the FAST fine-tune; `z-ft-lib` primary; LoRA first; closed loop at k = 4 and 10.

Also decided (2026-11-02): decoder heads `raw` and `dct`, selected per size offline (§3); the big-model side stays a linear projection (a learned remote encoder would move the decoding work remote, so the floor would measure where the cut is; possible later variant next to queue #6); no transformer / diffusion / flow decoders (EXP-0001 A1-arch check: transformer + L1 37.0% vs GRU + L1 51.9%, flow +0.6 pp). H1 thresholds: 100k, 0.90 / 0.70 of the 1M decoder.

Precondition: the 1M decoder ≥ 50% of `slow-only` (confirmed by the author 2026-11-02).

Recipe decided 2026-11-02 (§3): native formats, per-dataset quantile normalization, prompt condition. Still open (stage 0): the final subset list and the LIBERO weight, after checking how the LeRobot `pi05` code handles prompts, padding and normalization. Target format, verified in the installed packages: robosuite `OSC_POSE` with `control_delta: true` at 20 Hz (`libero/envs/env_wrapper.py`), 7-d action = delta position (3) + delta rotation (3, axis-angle) in [−1, 1], scaled to at most ±0.05 m and ±0.5 rad per step (`robosuite/controllers/config/osc_pose.json`), gripper −1 open / +1 close (our demos: exactly ±1).

## 10. Results

_Not started._

## Change log

- 2026-11-01: initial draft. Framing (the size of a latent → plan decoder as the floor of the on-robot model; a VLM latent, not the teacher's, because the teacher's plan is a linear readout of its hidden state) by the author. Builds on EXP-0001 §3 (backbone, FAST fine-tune, R1); the student-with-sensors stage is split off (§8).
- 2026-11-02 (before any run): author decided Open X-Embodiment (+ LIBERO) for the fine-tune, `z-ft-lib` as primary latent (was `z-ft-div`), LoRA first, closed loop at k = 4 and 10 (primary criterion at k = 4).
- 2026-11-02 (before any run): decoder heads `raw` and `dct` (fixed inverse DCT, n_c fixed from demo reconstruction before training), head and d_z selected per size on validation L1; big-model side stays linear. H1 thresholds (100k, 0.90 / 0.70) approved by the author.
- 2026-11-02 (before any run): precondition (1M decoder ≥ 50% of `slow-only`) confirmed by the author; all criteria now fixed. LIBERO target action format recorded in §9.
- 2026-11-02 (before any run): fine-tuning recipe fixed by the author: native action formats with per-dataset quantile normalization and a robot / control-mode / rate condition in the prompt (π0.5 style), gripper sign aligned, contradictory datasets excluded, LIBERO up-weighted. Evidence: `cross_embodiment_brief.md`.
- 2026-11-02 (stage 0, LeRobot code check; huggingface/lerobot @ f5c9bc0, v0.6.2, read only, not installed). Findings that affect §3, decisions pending with the author:
  1. `pi05` trains with the flow-matching loss only (`policies/pi05/modeling_pi05.py:932`, MSE; `knowledge_insulation` only detaches VLM K/V); there is no FAST-token loss. FAST-token training exists as a separate policy, `pi0_fast` (PaliGemma `gemma_2b` + `lerobot/fast-action-tokenizer`, `max_action_tokens` 256).
  2. LeRobot requires `torch>=2.7,<2.12` (`pyproject.toml:62`); the project pins 2.14.1+cu126. The VLM stage needs its own environment.
  3. Multi-dataset training is disabled (`datasets/factory.py:202`: "The MultiLeRobotDataset isn't supported for now"), and `MultiLeRobotDataset` would merge stats into one set (`compute_stats.aggregate_stats`), so per-dataset quantile normalization needs our own step (normalize each dataset with its own quantiles, merge, train with identity normalization).
  4. Prompt format: `Task: {task}, State: {256-bin discretized normalized state};\nAction: ` (`policies/pi05/processor_pi05.py:88`); state is required; state and action padded to 32 dims (`configuration_pi05.py:57–58`); normalization `QUANTILES` for both (L95–96). The robot / control / rate condition can go into the task text.
- 2026-11-02 (stage 0, option (a), author: look into it): `pi05_base`'s VLM weights into LeRobot's `pi0_fast` (FAST-token training). From the `lerobot/pi05_base` safetensors header (812 tensors, rev 22c30e5b, F32; read by HTTP range, not downloaded) and the `pi0_fast` code (@ f5c9bc0): names and shapes match. `pi0_fast` builds plain PaliGemma (`use_adarms[0] = False`, `modeling_pi0_fast.py:235`), so the LM layers are standard (`input_layernorm.weight` etc.), as in `pi05_base` (18 layers, width 2048, MLP 16384, 1 KV head × 256, vocab 257,152; `pi0_fast` L119/129); `from_pretrained` adds the `model.` prefix and copies `lm_head` into `embed_tokens` (L871–882, L918–935). `pi05_base`'s action-expert tensors (`gemma_expert.*`, `action_in/out_proj`, `time_mlp_*`) are unused, so loading needs `strict=False` and an empty `missing_keys` check. Not verified yet: an actual load; whether `pi05_base`'s `lm_head` was trained on FAST tokens with the same vocab mapping as `pi0_fast` (`fast_skip_tokens`, L1042–1052); whether `lerobot/fast-action-tokenizer` is FAST+. Fallback if (a) fails: the author's FAST+ implementation for SmolVLA, used to build a `pi05_fast`.
- 2026-11-03 (stage 0, load check, job 125173, `results/load_check.json`): **option (a) works.** VLM env built (job 125170: torch 2.11.0+cu128, LeRobot 0.6.2 from the f5c9bc0 source archive, transformers 5.5.4; a git dependency had failed because the HPC git config requires git-lfs, absent on compute nodes). (1) PaliGemma tokenizer reachable (needed a fine-grained HF token with gated-repo access; fixed by the author). (2) `pi05_base` → `pi0_fast`: all 604 model tensors filled (603 from the checkpoint + `embed_tokens` from `lm_head`), 0 missing, shapes match, the 209 unused tensors are all pi05's action expert. (3) FAST+ (`lerobot/fast-action-tokenizer`, byte-identical to `physical-intelligence/fast`) on 512 LIBERO 1 s chunks (20 steps, per-dimension 1 / 99% quantiles): 32.2 tokens per chunk on average (max 61; limit 256), reconstruction L1 0.009 in action units (0.017 normalized; max 0.126 normalized), tokens map to PaliGemma ids 254,979–256,920 (vocab 257,152, 128 skipped). Still open: whether `pi05_base`'s `lm_head` already predicts FAST tokens in this mapping (teacher-forced loss on LIBERO frames vs a uniform baseline; needs images).
- 2026-11-03 (stage 0, FAST loss test, jobs 278655/278656 on h200, `results/fast_ce.json`): **`pi05_base`'s LM head does not predict FAST+ tokens in `pi0_fast`'s mapping.** Teacher-forced `pi0_fast` loss on 64 LIBERO frames (openpi preprocessing: 180° images at 224, 8-d state, 1 s chunks, per-dimension quantiles; 39 action tokens per chunk on average incl. fixed prefix / end tokens): `pi05_base` 15.92 nats per token with the true chunks, 15.78 with chunks shuffled across frames, plain PaliGemma (`google/paligemma-3b-pt-224`, vocab rows sliced 257,216 → 257,152) 20.59; uniform over the full vocab 12.46, over the 2048 FAST tokens 7.62. Both models are worse than uniform and `pi05_base` does not use the observation (true ≈ shuffled), so the FAST head is learned in the fine-tune. FAST+ ids fall in PaliGemma's reserved `<loc>` / `<seg>` range (ids from 256,000; FAST ids 254,979–256,920), which may explain the confident wrong predictions (reader-inferred). `pi05_base` stays the starting point (robot-pretrained VLM; lower loss than plain PaliGemma); the debug LoRA run shows how fast the loss falls.
- 2026-11-03 (stage 0, mixture pipeline debug build, job 278756 on calc-g-002 CPU-only, author's choice; 20 episodes each of LIBERO and `utaustin_mutex` → `$ECVLA_SCRATCH/exp0005/datasets/mix_debug`, 6,860 frames, 300 MB): the builder works (per-dataset quantiles, condition in the task text, one LeRobot dataset, `mixture_stats.json`). Images checked visually: LIBERO upright after the 180° flip; mutex videos decode (torchcodec cannot load FFmpeg on the node, LeRobot falls back). **Gripper conventions from the data:** LIBERO +1 = close (commanded +1 goes with the closed finger, corr 0.68); mutex 0 / 1 with **1 = open** (commanded 1 goes with a wide opening, corr 0.67), so `grip_sign = −1` for mutex. Each new source gets this check before use. The debug build's 20 LIBERO episodes are all from one `libero_object` task (file order); the full build takes all.
- 2026-11-03 (stage 0, debug LoRA fine-tune, job 278763 on h200, ~14 min): `pi0_fast` from the `pi05_base` start checkpoint (`scripts/exp0005_make_init_ckpt.py`, with pre/post-processor pipelines), LoRA r 16 on the LM's attention + MLP, LM head trained in full, vision frozen; `mix_debug` (6,860 frames), batch 16, 500 steps, seed 0, fp32. Training loss (nats per action token): 14.13 (step 10), 10.53 (50), 7.18 (100), 5.92 (150), 5.04 (300), 4.95 (500); below uniform over the full vocab (12.46) by ~step 40 and over the FAST tokens (7.62) by step 100. So the FAST head is learned quickly with this setup. 1.08 s per step, 113 GB GPU memory (fp32 + Adam on the 527M-parameter LM head; bf16 for the full run). Training loss on 1.2 epochs of a tiny mixture: shows that it learns, not policy quality. LR fell 1.8e-5 → 2.5e-6 over the 500 steps although the configured warmup is 1,000 steps (LeRobot appears to rescale the schedule to the run length; not verified). Checkpoint: `$ECVLA_SCRATCH/exp0005/runs/debug_lora_500/checkpoints/000500`.
- 2026-11-03 (stage 0, slow-only sim check, job 278781 on h200, `results/slow_only_debug_lora_500.json`): the loop runs end to end (debug model, 2 episodes on the one trained task `pick_up_the_alphabet_soup_and_place_it_in_the_basket`): adapter loaded on its base (LM head changed, max 0.006), 280 steps each, 14 decoded 1 s chunks, both failed (expected after 500 steps on 20 demos); actions vary (arm std 0.5–1.2 raw, gripper opens and closes). Decode 1.8 s per 1 s chunk (fp32, H200): too slow for real-time `slow-only`; bf16 to try (the R1 latent itself is one forward pass). **Timing offset in the LIBERO HDF5s:** the stored obs at frame t (`ee_pos`, `ee_ori`, `gripper_states`) match the sim state `states[t + 1]` (max error 0.4 mm / 0.0006 rad / 0), not `states[t]` (9.7 mm / 0.014 rad), so the axis-angle conversion is right but the HDF5 obs (and presumably images) are one step (50 ms) after the state the action `actions[t]` was taken in. Our mixture pairs obs[t] with actions[t]; in closed loop the policy sees the current obs. Open: shift the pairing (obs[t] with actions[t + 1]) or re-render obs from `states` (as the EXP-0001 pipeline replays them); author to decide. Also in this step: the LeRobot `utaustin_mutex` task texts contain `tf.Tensor(b"…")` wrappers and literal `\n`; the builder now cleans them and puts the condition first (`robot: …; control: …; … Hz; task: …`), because `pi0_fast` cuts the prompt at 200 tokens and the mutex instructions are long.
- 2026-11-03 (before the full fine-tune): control-rate handling decided by the author (native steps + rate in the prompt, §3).
