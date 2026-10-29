# DRAFT: EXP-0001 amendment, A1-teacher v2 (not pre-registered yet)

- **Status:** draft for the author's review, 2026-10-29. Becomes active only when copied into the EXP-0001 README (§2, §7.1, Change log) **before** any of these runs start, and after the current H4 results are in.
- **Why:** the first A1-teacher results (Spatial task 0) show a 100k student at ~90% with the teacher's plan at k ≤ 4, collapsing at k = 10 and 20. Two causes are mixed in that curve, and one idea is untested:
  1. **Chunk-length confound.** With period k and delay k, the plan is k to 2k−1 steps old, but the teacher's chunk covers 8 steps. At k ≥ 10 the plan has run out before it arrives. The current curve measures the chunk length, not the student.
  2. **The student must learn delay compensation and chunk indexing itself**, from an age scalar. That is hard for a 10k network and may explain its seed variance (24% vs 78% on task 0).
  3. **Trust in z only depends on its age.** A stale plan is equally wrong whether 3 or 30 steps old if the world changed (e.g. the object slipped out of the gripper). The fast sensors can detect that; age cannot.

## Mechanisms (from Kalman filtering and control)

| ID | Mechanism | Kalman / control analogue | Change |
|---|---|---|---|
| **L** | Long teacher chunk: H = 32 steps (1.6 s) | Prediction horizon covers the measurement delay | Train 3 new 10M `task-id` teachers per suite with H = 32 (training only; the 8-step students stay as V0) |
| **A** | Time-aligned plan: feed `plan[age : age + 8]` instead of the raw chunk + age | Delay compensation (Smith predictor) | Input preprocessing; age still given |
| **R** | Residual action head: action = `plan[age]` + learned correction (correction initialised to 0) | Feedforward + feedback; tube MPC (nominal plan + ancillary controller) | Head change; at init the student reproduces the teacher's plan |
| **P** | Rewind-and-replay: train with z applied at the frame it was computed from (delay 0, period k); at run time, when z from step s arrives at t, recompute the core from s to t with z in place, using buffered fast inputs | Out-of-sequence measurement update (re-filtering); netcode reconciliation | Training uses delay 0; the closed loop keeps a 2k-step input buffer |
| **G-age** | Trust in z decays with age: attention bias −α·age on the z token (α learned per head) | Kalman gain with covariance growing between updates | One parameter per head |
| **G-inn** | Trust in z follows the innovation: a forward head predicts the next step's torque and F/T windows from (aligned plan, proprio); the normalized prediction error (NIS-like, running over the last 3 steps) biases attention on z: −β·NIS | Innovation gating (χ² gate) | Forward head trained with an auxiliary loss on demo data (normal data only) |
| **E** | Event-triggered query: when NIS exceeds a threshold fixed on validation demos (99th percentile), query the slow model now instead of waiting for the next period; the result still arrives k steps late | Event-triggered estimation | Closed-loop only |

## Variants (ablation ladder, each adds one mechanism)

| Variant | Teacher chunk | A | R | P | Gate | E |
|---|---|---|---|---|---|---|
| V0 (current A1-teacher) | 8 | – | – | – | none (age input) | – |
| V1 | 32 | – | – | – | none (age input) | – |
| V2 | 32 | ✓ | ✓ | – | none | – |
| V3 | 32 | ✓ | ✓ | ✓ | none | – |
| V4-age | 32 | ✓ | ✓ | ✓ | G-age | – |
| V4-inn | 32 | ✓ | ✓ | ✓ | G-inn | – |
| V5 | 32 | ✓ | ✓ | ✓ | G-inn | ✓ |

Students: blind, 10k and 100k, 3 seeds, Spatial + Object (Goal optional). z = action chunk. k ∈ {1, 4, 10, 20}.

## Perturbation evaluation (eval-only, new)

**Grasp loss:** in closed loop, once the gripper has held the task's target object for 10 consecutive steps, the object is moved 5 cm below the gripper's fingers and released (one event per episode, at a seeded random time within the carry phase). Success = task success by the horizon. The same 50 initial states and seeds for every variant; the teacher alone and 10M `task-id` are run under it as references.

The demos contain no recoveries, so absolute recovery rates will be low for every policy; the comparison between gates is what matters.

## Hypotheses and criteria (to fix before running)

| ID | Hypothesis | Supported if | Refuted if |
|---|---|---|---|
| **H4b** staleness is the chunk, not the student | With a plan that covers the delay, a tiny student tolerates slower updates | V3 10k holds ≥ 50% of its teacher's success at k = 10 (Spatial + Object pooled), where V0 10k does not | V3 10k at k = 10 is within 10 pp of V0 10k at k = 10 |
| **H4c** structure substitutes for size | Alignment + residual make the 10k student closer to the teacher | At k = 1, V2 10k − V0 10k ≥ 10 pp and the seed spread (max − min) of V2 10k is ≤ half that of V0 10k | V2 10k ≤ V0 10k |
| **H4d** innovation beats age for trust | Under grasp loss, trust driven by the fast sensors helps; age alone cannot see the event | Under grasp loss at k = 10: V4-inn − V4-age ≥ 10 pp **and** unperturbed V4-inn ≥ V4-age − 3 pp | V4-inn ≤ V4-age under grasp loss |
| **H4e** (exploratory) event-triggered queries | Re-querying on an innovation spike speeds recovery | Reported: V5 − V4-inn under grasp loss, and queries per episode | – |

Replay (P) has no own hypothesis: V3 vs V2 is reported. P makes one student usable at any delay, which is the property P-onboard vs P-remote needs.

## Cost (estimate)

- Teachers: 3 seeds × 2 suites × 10M GRU with H = 32 ≈ 6 GPU runs × ~20 min.
- Students: 7 variants × 2 sizes × 3 seeds × 2 suites = 84 runs (CPU for 10k with 8-d inputs; GPU for the rest), ~25–40 min each.
- Evals: 84 runs × 4 k = 336 jobs (~1.5 h each, CPU), plus grasp-loss evals for V3–V5 and the references at k ∈ {1, 10}: ~80 jobs.

## Open choices for the author

1. NIS window (3 steps) and the E threshold (99th percentile on validation demos).
2. Whether the grasp-loss perturbation is the right first event, or a mid-carry object push instead.
3. Whether to drop V1 (long chunk without alignment) to save ~15% of the cost.
