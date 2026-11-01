# EXP-0005: How small can an on-robot decoder be that reconstructs the plan from a VLM latent?

- **Status:** draft (2026-11-01), awaiting the author's decisions (§9) and approval of the criteria
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

- **Backbone** as EXP-0001 §3: `pi05_base` (LeRobot port), fine-tuned autoregressively on FAST+ action tokens, then **frozen**. One backbone run serves the whole grid.
  - Primary latent `z-ft-div` (diverse robot data, no LIBERO; author's decision of 2026-10-28). `z-ft-lib` (adds the LIBERO training demos of the evaluated suites) is the secondary variant. `z-frozen` is not run (§9).
- **Readout R1** (EXP-0001 §3): final-layer hidden state at the position that predicts the first action token, concatenated with the mean-pooled prefix. One forward pass, no decoding. Inputs: agent-view + wrist RGB and the instruction.
- **Slow-side projection.** A linear map from R1 to d_z ∈ {16, 64, 256}, trained jointly with each decoder but **run on the slow machine** (it is part of what is transmitted, and it sets the payload). Not counted in the on-robot size; its parameters and the payload bytes are reported.

### On-robot decoder (the variable)

- MLP: d_z → hidden → hidden → 40 × 7 (a 40-step chunk, as the EXP-0002/0003 teachers), GELU, L1 loss on normalized demo actions (same normalization and target construction as `train.py`). Input is z only: no proprio, no task token, no age (fresh plan, age 0; delay enters only in closed loop, below).
- **Sizes (on-robot parameters):** 3k, 10k, 30k, 100k, 300k, 1M (hidden width set per size and d_z). For each size, d_z is chosen by validation L1 on held-out demos (never on closed-loop success).
- **Data:** LIBERO-Spatial and -Object demos (`prepared_v2`), same train / held-out split as `train.py` (last 2 kept demos per task held out). z precomputed once for all demo frames.
- **Seeds:** 3 per (size, d_z), one backbone.

### Closed loop

- **`decoder-replay`:** the VLM (frozen) is queried every k = 4 steps from the current observation; its z arrives exactly k steps later; the decoder turns it into a 40-step plan, and the robot executes step `age` of the latest delivered plan (as EXP-0003's `replay`). Before the first plan: hold start as EXP-0004 (first plan treated as fresh), to keep start-up effects out.
- k = 4 only (the project's operating point: ~5 Hz slow updates, 20 Hz control). Evaluation as EXP-0001–0004: 50 fixed initial states × 10 tasks per suite, eval seed 0.
- **References** (same protocol): `slow-only` (the fine-tuned backbone decodes its own FAST action chunk, executed the same way at k = 4) and the **1M decoder** (the reference in §5).

### Positive control (offline, cheap)

The same decoder grid on the EXP-0002 H40 teacher's hidden state (`z_hidden`, 704-d, already precomputed), offline L1 only. Its plan is a linear readout of that state, so small decoders should match large ones up to the d_z bottleneck. A failure here points at the pipeline, not the latent.

## 4. Measurements

- **Offline:** plan L1 on held-out demos per chunk step (as EXP-0002 post-hoc), per (latent, size, d_z, seed); the VLM's own decoded FAST chunk (`slow-only`) on the same frames as a reference.
- **Closed loop:** success per (suite, size, seed) at k = 4; pooled Spatial + Object per seed; ratio to the 1M decoder (mean / mean) with bootstrap 95% CI over seeds; per-seed values.
- **The size curve** (success vs on-robot parameters) is the main reported result.
- Payload (d_z × bytes) per update; slow-side projection parameters; R1 width.

## 5. Pre-registered criteria

Primary latent `z-ft-div`, closed loop at k = 4, Spatial + Object pooled, 3 seeds.

- **Precondition (H0-latent):** the 1M decoder reaches ≥ 50% of `slow-only`'s success. If not, the result is "the latent does not carry a usable plan" and H1 is not evaluated; `z-ft-lib` is then reported as the in-domain check.

| | Supported if | Refuted if |
|---|---|---|
| **H1, primary** | some decoder ≤ 100k reaches ≥ 0.90 of the 1M decoder's success | no decoder ≤ 100k reaches ≥ 0.70 of it |

In between: inconclusive. **Why 100k and 0.90 / 0.70:** 100k is the size at which students matched their teacher from a teacher plan (EXP-0001/0002); the question is whether that survives a latent that must be decoded. The thresholds are proposals for the author (§9).

Reported, not gating: the smallest size reaching 0.90 (the floor); `z-ft-lib`; `slow-only` vs the 1M decoder; offline L1 vs closed-loop success across sizes; the positive control.

## 6. Threats to validity

- **One backbone run.** Seeds vary the decoder only; backbone variance is not measured.
- **`pi05_base` pretraining may include LIBERO** (EXP-0001 §5 risk). Check its data card before the backbone stage; if it does, `z-ft-div` is partly in-domain. Logged either way.
- **The slow-side projection does real work.** Counting it as "remote" is a deployment choice (it runs where the backbone runs and cuts the payload). Reported separately so the on-robot floor can be read with and without it.
- **Readout R1 only.** A poor floor could be the readout; R2 (decoded-token states) is a later variant.
- **LIBERO needs no scene state** (EXP-0001 A-test), so the latent only has to carry the plan. The floor measured here is a floor for plan decoding, not for state extraction.
- **No robot sensors in the decoder** (by design; §8). The closed loop is the decoded plan executed open loop between updates, as `replay`.
- **Action-space mapping.** FAST+ tokens from diverse data must map consistently onto LIBERO's delta-pose controller (EXP-0001 §8 open item).

## 7. Plan and cost

0. **Infrastructure (prerequisite, not a test):** LeRobot `pi05_base` in the project env (check against the pinned torch 2.14.1+cu126); diverse dataset (§9); FAST fine-tune → `z-ft-div` (then `z-ft-lib`) on H200; `slow-only` LIBERO success as a sanity check. Debug first: a few hundred steps on `debug` / `h200`, a 2-episode `slow-only` eval.
1. Precompute R1 for all Spatial + Object demo frames (one forward pass per frame, GPU).
2. Positive control and offline decoder grid (6 sizes × 3 d_z × 3 seeds × 2 suites; small MLPs on CPU).
3. Closed loop at k = 4: 6 sizes × 3 seeds × 2 suites plus `slow-only` (GPU: the VLM runs in the loop every 4 steps). Cost estimated from the debug eval before submitting.
4. Analysis script written before any closed-loop run.

The backbone stage (0) is the expensive and uncertain part; its cost is estimated in the debug run and logged before the full fine-tune.

## 8. Out of scope (next experiment)

Starting the on-robot student from the smallest working decoder and adding the fast sensor inputs (proprio, torque, F/T, and later camera), compared against the same student trained from scratch. Its value can only show on tasks where the plan alone is not enough (EXP-0003/0004: on LIBERO the student mostly executes the plan), so that experiment also needs such a task.

## 9. Open decisions (author)

1. **Diverse dataset** for the FAST fine-tune (EXP-0001 §8, still open): which mixture (e.g. DROID / Open X-Embodiment subsets), and how its action spaces map onto LIBERO's controller.
2. **Primary latent:** keep `z-ft-div` (no LIBERO, decided 2026-10-28), or make `z-ft-lib` primary for this question (decoding a LIBERO plan from a backbone that never saw LIBERO actions may fail the precondition for reasons unrelated to size).
3. **Fine-tuning recipe:** full fine-tune vs LoRA (EXP-0001 §8, still open).
4. **Criteria:** 100k as the primary size; 0.90 / 0.70 of the 1M decoder; the 50% precondition against `slow-only`.
5. **Closed-loop k:** k = 4 only (proposed), or also k = 10.

## 10. Results

_Not started._

## Change log

- 2026-11-01: initial draft. Framing (the size of a latent → plan decoder as the floor of the on-robot model; a VLM latent, not the teacher's, because the teacher's plan is a linear readout of its hidden state) by the author. Builds on EXP-0001 §3 (backbone, FAST fine-tune, R1); the student-with-sensors stage is split off (§8).
