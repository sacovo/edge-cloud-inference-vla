# EXP-0001: When Does the Remote Latent Have to Carry State?

- **Status:** designed, not started
- **Created:** 2026-10-27
- **Platform:** simulated manipulator (robosuite / LIBERO family)
- **Related:** gaps G1, G2 and G9 and threat T1 in [`literature/SYNTHESIS.md`](../../literature/SYNTHESIS.md); replaces the original E1 grid.

The hypotheses, conditions, metrics and success criteria below are fixed **before** any run. Changes after the first result must be logged in the [Change log](#change-log) with a reason.

---

## 1. Question

A large fast policy may need the MLLM latent only as a **task identifier**. As the fast policy shrinks, or loses its own vision, it can no longer get scene state from its own sensors. At that point the latent has to **carry state**, and a task-ID substitute should stop working. Where is that crossover, and does sensitivity to latent staleness appear with it?

## 2. Hypotheses

- **H1, crossover.** For sighted fast policies, a task-ID embedding matches the refreshed MLLM latent at large head sizes and falls behind at small head sizes.
- **H2, blind policies depend on state in z.** For a blind fast policy (no vision input), the refreshed latent beats both the task ID and the latent held from the episode start, at every head size.
- **H3, staleness scales with dependence on state.** Success falls faster with the latent update period when the policy is smaller or blind than when it is large and sighted.
- **H0, null (the OpenHelix concern).** z carries little usable scene state, so the refreshed latent never beats the episode-start latent by a meaningful margin, at any size or vision level.

## 3. Design

A full factorial grid over four axes. Each cell is trained with 3 seeds.

| Axis | Levels |
|---|---|
| **A. Local vision** | `none` (blind) · `tiny` (32×32 grayscale wrist or agent-view camera, 5 Hz) · `full` (standard RGB, e.g. 128×128, 20 Hz) |
| **B. Head size** (trainable params excl. vision encoder) | ~10k · ~100k · ~1M · ~10M · ~50M |
| **C. Conditioning** | see table below |
| **D. Latent update period** | every 1 · 4 · 10 · 20 control steps (at 20 Hz control: 50 ms to 1 s) |

Axis D applies only to the conditions that are updated during the episode (`z-refresh`, `oracle-delayed`).

### Conditioning conditions (axis C)

| ID | Conditioning input | Purpose |
|---|---|---|
| `none` | nothing | Lower bound |
| `task-id` | Frozen embedding of the instruction (or one-hot task) | "z = task ID" |
| `z-ep0` | MLLM latent from the first frame, held for the whole episode | Task + initial scene, no updates |
| `z-refresh` | MLLM latent recomputed every *k* steps (axis D), delivered with *k*-step delay | Can carry current state |
| `oracle-delayed` | Ground-truth object poses + contact flags from the simulator, same *k* and delay as `z-refresh` | Upper bound on usable state under the **same** staleness; tells "z lacks state" apart from "state too old" |
| `oracle-fresh` | Ground-truth state, no delay | Absolute upper bound |

### Fast-policy inputs (all conditions)

| Signal | Rate |
|---|---|
| Joint positions/velocities, gripper state | 20 Hz |
| Joint torque | 60 Hz |
| Wrist F/T | 100 Hz |
| Camera (per axis A) | — |

Slow inputs are held at their latest value, the "hold latest" alignment (approach A in Theme 4), with update periods and latency randomized during training (approach B). Actions are output at 20 Hz.

### Fast-policy architecture

A single family scaled by width and depth:
- one encoder per modality
- a recurrent core (GRU) running at 20 Hz
- conditioning injected by FiLM, plus one cross-attention token for z
- an action-chunk head (chunk length 8, executed with re-planning every step)

The vision encoder is a small CNN for `tiny` and a frozen pretrained encoder plus trainable projection for `full`. Its parameters are reported separately from the head size.

### The slow model (z)

- **Model:** a ~27B open VLM. Candidates: Gemma 3 27B, Qwen2.5-VL-32B. The choice is open (see §8).
- **Inputs:** agent-view + wrist RGB and the instruction.
- **Latent:**
  - **Stage 2 (frozen):** a mean-pooled final-layer hidden state, then a trainable linear projection to d = 256. Only the projection trains with the fast policy.
  - **Stage 3 (adapted):** K = 8 learnable query tokens with LoRA on the VLM, trained end-to-end with the fast policy, so z can learn to carry what the head needs.
- **Delay model:** z computed from observation *t* becomes available at *t + k*. Training samples *k* from the same set as evaluation, plus jitter of ±1 step.

## 4. Tasks

Four task groups, ordered by how fast the scene state changes. Each group gets 3–4 tasks.

| Group | Example tasks | Scene dynamics | Expected blind result |
|---|---|---|---|
| **T-static** | LIBERO-Object / -Spatial pick-and-place, unchanged | Static after t=0 | Near sighted |
| **T-contact** | Peg insertion, drawer opening, wiping (robosuite) | Static objects; contact needs F/T | Near sighted if z localizes the object |
| **T-event** | Object displaced once mid-episode, injected grasp failure, distractor appears | One discrete change | Drops if the update period exceeds the reaction window |
| **T-dynamic** | Target moves at a constant speed (robosuite conveyor-style) | Continuous motion | Breaks down beyond some latency |

LIBERO tasks are mostly static. T-event and T-dynamic need custom robosuite variants built with scripted perturbations. Their demonstrations come from scripted or MimicGen-style generation (open decision, §8).

## 5. Measurements

- **Primary:** task success rate. 50 evaluation episodes per task × cell × seed, with fixed evaluation initial states (held out, never edited).
- **Derived gaps**, computed per (vision, head size, task group):
  - **Task value:** `SR(z-ep0) − SR(task-id)`
  - **State value:** `SR(z-refresh, k) − SR(z-ep0)`
  - **State efficiency of z:** `SR(z-refresh, k) / SR(oracle-delayed, k)`
- **Probes on z:**
  - linear-probe R² for object positions and gripper–object distance, using frozen z on held-out frames
  - within-episode variance of z relative to between-task variance

  These test H0 directly.
- **Also logged:** fast-policy FLOPs per step and parameter counts (head and vision separately); z dimension and payload size per update.

Statistics: mean ± 95% CI over seeds (bootstrap over episodes within each seed, then over seeds).

## 6. Pre-registered success criteria

pp = percentage points.

| Hypothesis | Supported if | Refuted if |
|---|---|---|
| **H1** | For `full` vision on T-event + T-dynamic: \|SR(z-refresh) − SR(task-id)\| ≤ 5 pp at the 50M head, **and** SR(z-refresh) − SR(task-id) ≥ 15 pp at the ≤100k head (CIs non-overlapping) | The gap is about the same (within 5 pp) at all head sizes |
| **H2** | For `none` vision: SR(z-refresh, k≤4) > SR(z-ep0) and > SR(task-id) by ≥ 10 pp on T-event + T-dynamic, at ≥ 4 of 5 head sizes | Fewer than 2 head sizes show the 10 pp margin |
| **H3** | The fitted slope of SR vs. log(k) is steeper (more negative) for `none` than for `full` vision at the same head size, in ≥ 4 of 5 head sizes, on T-event + T-dynamic | Slopes are indistinguishable (CIs overlap) at ≥ 3 head sizes |
| **H0** | State value ≤ 5 pp everywhere **and** probe R² < 0.3 for object position | Any H2 criterion is met |

If H0 holds for frozen z (Stage 2) but `oracle-delayed` shows a large state value, that is a positive finding about frozen VLM features, not a failed experiment. In that case, run Stage 3.

## 7. Stages

Each stage gates the next.

0. **S0: Infrastructure.**
   - an environment wrapper with multi-rate observations (robosuite `sampling_rate`), latency/jitter injection for z, and the T-event/T-dynamic variants
   - the demonstration pipeline
   - a smoke test (2-minute train + eval) and unit tests for the delay wrapper
1. **S1: Oracle-only pilot (no VLM).** Run conditions `none`, `task-id`, `oracle-delayed` and `oracle-fresh` across the whole vision × size × k grid. This is cheap and tests whether the task suite needs scene state at all. **Gate:** `oracle-fresh` beats `task-id` by ≥ 20 pp for blind policies on T-event + T-dynamic. If not, redesign the tasks before spending VLM compute.
2. **S2: Frozen 27B latent.**
   - Precompute z for all demonstration frames offline.
   - In closed-loop evaluation, run the VLM in the loop with simulated delay.
   - Add `z-ep0` and `z-refresh`, plus the probes.
3. **S3: Adapted latent (conditional).** Run Stage 3 if S2 shows H0 for frozen z, or a large gap between `z-refresh` and `oracle-delayed`.

## 8. Open decisions (resolve before S0 ends)

- Slow VLM: Gemma 3 27B vs Qwen2.5-VL-32B (licence, vision quality, ease of hidden-state access).
- Demonstration source for T-event/T-dynamic: scripted controller vs MimicGen-style augmentation of LIBERO demos.
- Camera resolution for `full` (LIBERO default vs 128×128), after confirming LIBERO/robosuite control and camera defaults.
- Compute budget and where VLM inference runs during closed-loop evaluation.
- Whether to add a mobile-manipulator variant (RoboCasa, with base IMU) as a follow-up. Not part of this experiment.

## 9. Threats to validity

- **Size confounded with pretraining.** All heads are trained from scratch with the same recipe. The `full` vision encoder is shared and frozen across head sizes.
- **Demonstrations limit what a blind policy can learn.** If the demos never show recovery from a perturbation, every condition fails on T-event. Demos must include recovery behaviour.
- **The oracle's state representation advantages it** (exact poses vs. a learned latent). That is intended: it is an upper bound, not a competitor.
- **The fast policy can infer the delay** from proprioception. Report results both with and without an explicit latent-age input (that variant belongs to E2 but is cheap to add here).

## 10. Results

_Not started._

## Change log

- 2026-10-27: initial design.
