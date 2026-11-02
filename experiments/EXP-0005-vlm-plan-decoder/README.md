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
