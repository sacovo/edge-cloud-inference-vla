# EXP-0001: When Does the Remote Latent Have to Carry State?

- **Status:** designed, not started. Phase A (LIBERO) first
- **Created:** 2026-10-27
- **Platform:** simulated manipulator (robosuite / LIBERO family); slow model either onboard (same Jetson) or remote; see §3 Deployment profiles
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

- **Model:** two latent sources, compared at matched conditions.
  - **`pi05_base`**: the PaliGemma-based VLM backbone of π0.5 (~3B, robot-pretrained), from openpi. We use the base checkpoint only, **not** `pi05_libero`: that one was fine-tuned on LIBERO demos and would make z look strong for in-domain reasons.
  - **PaliGemma 2 28B** (Gemma 2 27B + SigLIP-So400m, arXiv 2412.03555): generic, not robot-pretrained, and matching the ~27B remote-model budget.
  - Optional secondary sweep: PaliGemma 2 3B / 10B / 28B. These share an image encoder, so this varies the slow model's size and nothing else.
  - Weights are under Gemma terms (openpi code is Apache-2.0). Check them before redistributing any derived weights.
- **Inputs:** agent-view + wrist RGB and the instruction.
- **Latent:**
  - **Stage 2 (frozen):** a mean-pooled final-layer hidden state, then a trainable linear projection to d = 256. Only the projection trains with the fast policy.
  - **Stage 3 (adapted):** K = 8 learnable query tokens with LoRA on the VLM, trained end-to-end with the fast policy, so z can learn to carry what the head needs.
- **Delay model:** z computed from observation *t* becomes available at *t + k*. Training samples *k* from the same set as evaluation, plus jitter of ±1 step.

### Deployment profiles

The research question does not depend on *where* the slow model runs. The deployment only changes the latency distribution of z, so each profile is a delay model, not a separate experiment.

| Profile | Slow model | Where | Latency model for z | Why |
|---|---|---|---|---|
| **P-onboard** | `pi05_base` (~3B) | Same Jetson as the fast policy, in a separate slow loop (≈1–4 Hz) | Fixed compute latency, no loss; may compete with the fast loop for the GPU | The author's observation: a 3B VLA fits on a Jetson but cannot produce actions fast enough for control. As a slow loop it only has to emit z. Theme 3 reports π0.5 at 10.3 Hz on Jetson Orin after heavy optimisation (EagleVLA) and 1421 ms naive. Measure the backbone-only time ourselves. |
| **P-remote** | PaliGemma 2 28B | Stationary GPU machine over WiFi | Compute latency + network jitter, loss and outages | The original edge–cloud setup |

This gives a direct trade-off: a smaller, fresher latent (P-onboard) against a larger, staler one (P-remote). Measure the real slow-loop latency of each profile in S0 and set the axis-D update periods from it.

## 4. Tasks

Four task groups, ordered by how fast the scene state changes. Each group gets 3–4 tasks.

| Group | Example tasks | Scene dynamics | Expected blind result |
|---|---|---|---|
| **T-static** | LIBERO-Object / -Spatial pick-and-place, unchanged | Static after t=0 | Near sighted |
| **T-contact** | Peg insertion, drawer opening, wiping (robosuite) | Static objects; contact needs F/T | Near sighted if z localizes the object |
| **T-event** | Object displaced once mid-episode, injected grasp failure, distractor appears | One discrete change | Drops if the update period exceeds the reaction window |
| **T-dynamic** | Target moves at a constant speed (robosuite conveyor-style) | Continuous motion | Breaks down beyond some latency |

LIBERO tasks are mostly static. T-event and T-dynamic need custom robosuite variants built with scripted perturbations.

### Demonstrations: bootstrap with policy rollouts

1. **Seed:** teleoperate a few episodes per task in sim (robosuite teleop), with perturbations active for T-event/T-dynamic.
2. **Bootstrap policy:** train a sighted policy (ACT or SmolVLA) on the seed demos.
3. **Collect:** roll it out with perturbations and randomized initial states. Keep the **successful** episodes, with full multi-rate sensor logs and ground-truth state.
4. **Iterate:** retrain on seed + kept episodes. Add manual episodes where success stays low, until each task has enough demos. Optionally improve the collector with the author's RECAP-on-SmolVLA work.
5. **Freeze** the final dataset with a version tag before any EXP-0001 training. Evaluation initial states come from a held-out seed range and are never used for collection.

The collector is sighted, so blind students learn from a sighted teacher's trajectories (privileged-teacher style). Two biases follow from filtering on success:
- The kept episodes over-represent easy initial states. Log the distribution of kept initial states and report it.
- Recovery behaviour only appears where the collector actually recovered. Count the recoveries per T-event task, and add manual recovery demos if they are rare.

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

## 6. Pre-registered success criteria (Phase B)

These apply to Phase B (§7.4). Phase A has its own criteria in §7.2.

pp = percentage points.

| Hypothesis | Supported if | Refuted if |
|---|---|---|
| **H1** | For `full` vision on T-event + T-dynamic: \|SR(z-refresh) − SR(task-id)\| ≤ 5 pp at the 50M head, **and** SR(z-refresh) − SR(task-id) ≥ 15 pp at the ≤100k head (CIs non-overlapping) | The gap is about the same (within 5 pp) at all head sizes |
| **H2** | For `none` vision: SR(z-refresh, k≤4) > SR(z-ep0) and > SR(task-id) by ≥ 10 pp on T-event + T-dynamic, at ≥ 4 of 5 head sizes | Fewer than 2 head sizes show the 10 pp margin |
| **H3** | The fitted slope of SR vs. log(k) is steeper (more negative) for `none` than for `full` vision at the same head size, in ≥ 4 of 5 head sizes, on T-event + T-dynamic | Slopes are indistinguishable (CIs overlap) at ≥ 3 head sizes |
| **H0** | State value ≤ 5 pp everywhere **and** probe R² < 0.3 for object position | Any H2 criterion is met |

If H0 holds for frozen z (Stage 2) but `oracle-delayed` shows a large state value, that is a positive finding about frozen VLM features, not a failed experiment. In that case, run Stage 3.

## 7. Stages

The experiment runs in two phases. **Phase A** uses an existing suite with its existing demonstrations, so it can start without building a data pipeline. **Phase B** uses the bootstrapped event/dynamic tasks from §4 and the criteria in §6. Phase B starts only if Phase A passes its gate (§7.3).

### 7.1 Phase A: LIBERO with existing demonstrations

**Why LIBERO first:**
- It ships human demonstrations: 50 per task, 10 tasks per suite.
- It has the most published baselines.
- Its suites already differ in how much scene state a blind policy needs. Each suite pins down one factor and varies another, per the LIBERO paper:

| Suite | What varies across tasks | Expected need for scene state (blind policy) |
|---|---|---|
| LIBERO-Goal | Goal only (same objects, same layout) | **Low**: task ID should get a blind policy far |
| LIBERO-Spatial | Layout (same objects) | **High**: must locate the referenced object |
| LIBERO-Object | Target object (same layout) | **Medium–high** |
| LIBERO-10 (long) | Multi-step tasks | High; optional, run last |

LIBERO-Goal is a built-in control: if z helps a blind policy there as much as on Spatial, z is probably doing something other than providing state.

**Phase A grid** (reduced, to start sooner):

| Axis | Levels |
|---|---|
| Vision | `none`, `full` (`tiny` deferred) |
| Head size | ~10k, ~100k, ~1M, ~10M |
| Conditioning | all six from §3 |
| Update period | k ∈ {1, 10} |

That is 3 seeds per cell, evaluated on the suite's standard initial states.

**Data:** replay the LIBERO demonstrations in robosuite from their saved simulator states. Replaying regenerates the observations at our own sensor rates and adds joint torque, wrist F/T and ground-truth object poses, which the stored observations may not contain. That replay is feasible is an assumption; verify it in A0.

**Steps:**
- **A0: Infrastructure.**
  - demo replay at multiple sensor rates
  - the z delay/jitter wrapper
  - a check of LIBERO's control rate and camera defaults
  - the backbone-only `pi05_base` latency measurement
  - a smoke test and unit tests for the delay wrapper
- **A1: Oracle pilot (no VLM).** Conditions `none`, `task-id`, `oracle-delayed`, `oracle-fresh`.
- **A2: `pi05_base` latent.** Precompute z for all demo frames. Run closed-loop evaluation with z in the loop under the delay model. Add `z-ep0`, `z-refresh` and the probes.
- Check whether `pi05_base` pretraining data includes LIBERO. If it does, log it as a threat to validity: z would be partly in-domain.

### 7.2 Phase A pre-registered criteria

"Gap" = SR(condition) − SR(`task-id`), computed for blind (`none`) policies, on LIBERO-Spatial + LIBERO-Object pooled.

| Criterion | Pass if | Meaning if it fails |
|---|---|---|
| **A-test (the suite can test the idea)** | `oracle-fresh` gap ≥ 20 pp at ≥ 2 head sizes | LIBERO does not need scene state for blind policies. That is **not** evidence against the concept; go to Phase B directly |
| **A-z (z carries state)** | the `z-refresh` (k=1) gap reaches ≥ 50% of the `oracle-fresh` gap at ≥ 2 head sizes ≤ 1M | Frozen `pi05_base` z carries little usable state (H0 for this latent). Consider the adapted latent (S3) before Phase B |
| **A-control** | On LIBERO-Goal, the `z-refresh` gap is smaller than on Spatial + Object (CIs not overlapping) | z helps for reasons other than state. Investigate before Phase B |
| **A-crossover (indicative only)** | For `full` vision, \|SR(`z-refresh`) − SR(`task-id`)\| at the 10M head is smaller than at the 10k head | Reported, not gating; Phase A is too static for a clean H1 test |

### 7.3 Gate to Phase B

Start Phase B (bootstrapped event/dynamic tasks, §4 demonstrations, full grid, §6 criteria) if **A-z and A-control pass**. If A-test fails, Phase B is the only way to test the idea, so start it directly.

### 7.4 Phase B and the adapted latent

- **B0–B2:** the original S0–S2 plan on the T-static/contact/event/dynamic tasks. Gate: `oracle-fresh` beats `task-id` by ≥ 20 pp for blind policies on T-event + T-dynamic; otherwise redesign the tasks before spending VLM compute.
- **S3: Adapted latent (conditional):** run if A2 or B2 shows H0 for frozen z, or a large gap between `z-refresh` and `oracle-delayed`.

## 8. Open decisions (resolve before A0 ends unless marked Phase B)

- ~~Slow VLM~~ (resolved 2026-10-28): `pi05_base` + PaliGemma 2 28B; see §3.
- ~~Demonstration source~~ (resolved 2026-10-28): teleop seed + bootstrap rollouts; see §4.
- Phase B: collector policy (ACT vs SmolVLA), and whether a RECAP step is worth it.
- Measured slow-loop latency for P-onboard (backbone only, on the target Jetson) and P-remote.
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
- 2026-10-28 (before any run):
  - The slow model is now `pi05_base` (base checkpoint only, to avoid LIBERO in-domain skew) plus PaliGemma 2 28B, replacing the Gemma 3 / Qwen candidates.
  - Added deployment profiles: P-onboard (both loops on one Jetson) and P-remote.
  - Demonstrations now come from teleop seed episodes plus success-filtered policy rollouts.
  - No hypothesis, condition or criterion changed.
- 2026-10-28 (before any run): split into Phase A (LIBERO with existing demos, reduced grid, its own pre-registered criteria in §7.2) and Phase B (bootstrapped event/dynamic tasks, unchanged §6 criteria). Phase B is gated on Phase A, so we can start sooner and build our own dataset only if the concept shows up.
