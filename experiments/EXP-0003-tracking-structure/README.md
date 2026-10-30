# EXP-0003: Does a tracking structure let a small policy use an old plan?

- **Status:** running (pre-registered 2026-10-30; debug runs passed, full runs started 2026-10-30)
- **Platform:** LIBERO-Spatial and LIBERO-Object, blind fast policies, simulated delay (same setup as EXP-0002)
- **Related:** EXP-0002 §8 (H1 result and the post-hoc plan-error check), `experiments/QUEUE.md` (#3); literature Theme 7 Sec. 1 row R and Sec. 3 (residual feedback around a plan; age-indexed plans).

The hypothesis, conditions and criteria below are fixed **before** any run. Changes after the first result go in the Change log with a reason.

## 1. Question

In EXP-0002, a 100k student given the H = 40 teacher's plan reached 60.5% at update period k = 10 but only 11.6% at k = 20, although the plan still covers the oldest age used at k = 20 (39 steps). On demos, the plan's error grows only gradually with age (L1 0.100 at steps 10–19, 0.122 at steps 20–39; EXP-0002 post-hoc), so a plan 20–39 steps old still carries most of the information. The EXP-0002 student receives the whole 280-d chunk plus its age and has to find the relevant step itself, and it has no explicit way to correct the plan with what it senses.

Does a **tracking structure**, i.e. giving the student the plan step that is due now and letting it add a bounded per-step correction, let it use an old plan?

## 2. Hypothesis

- **H1 (tracking).** With an age-aligned plan window and a bounded per-step residual, the student keeps a useful share of its teacher's success at k = 20, where the EXP-0002 student collapsed.
- **H0 (null).** The structure does not help: at k = 20 the plan is too stale for any student to use, or the student cannot correct it.

A reference arm (the teacher executing its own delayed plan, no student) tells the two halves of H0 apart (§4).

## 3. Design

One change against EXP-0002: how the student receives and uses the plan.

| | EXP-0002 (baseline, reused) | EXP-0003 `track` (new) |
|---|---|---|
| Teacher, plan | 10M blind `task-id` GRU, H = 40 (EXP-0002 runs and latents, reused) | same |
| z given to the student | whole chunk (40 × 7 = 280) + age | **age-aligned window** `plan[age … age + 7]` (8 × 7 = 56), steps beyond 39 padded with step 39, + age |
| Student output | own 8-step chunk, first action executed | **`a_i = plan[age + i] + r_i`**, residual r bounded per step (below), zero-initialised; first action executed |
| Before the first plan arrives | z masked | z masked, base = 0, residual not clipped (acts as before) |
| Sizes, training delay, evaluation, seeds | 10k / 100k; k ∈ {1, 4, 10, 20}, delay k − 1..k + 1; 50 initial states × 10 tasks, delay exactly k; 3 seeds | identical |

- **Residual bound:** |r| ≤ 0.5 in raw action units on the 6 arm dimensions (position, rotation), as on LIBERO in [xiao2026selfimproving] App. B.2 (via Theme 7); the gripper residual is not bounded, because a stale plan's gripper timing is what degrades most (EXP-0002 post-hoc: gripper L1 0.154 → 0.247 from step 0 to steps 20–39) and a bound below 1 could never flip it. Bounds are applied in the normalized space via the action std.
- **Reference arm `replay`:** the H40 teacher alone executes `plan[age]` of its latest delivered plan (same delay model, k ∈ {1, 4, 10, 20}); before the first plan, it acts on its own current output. No training; 6 teachers × 4 k.
- **Baseline reuse.** EXP-0002 student numbers (`EXP-0002-plan-horizon/results/students_h40.json`) and teachers (`teachers_h40.json`): same data (`prepared_v2`), teachers, latents, split and evaluation protocol. Not re-run.
- **Code changes:** a policy option `track` (window 8, residual bound), the aligned window in training batches and in evaluation, and a `replay` evaluation mode for teacher runs. Defaults leave EXP-0001/0002 behaviour unchanged (tests: zero residual reproduces `plan[age]`; window indices and padding; stepwise = sequence; `track` off reproduces the EXP-0002 student's parameter count).

## 4. Measurements

- Success rate per (suite, arm, size, k), mean and range over 3 seeds; `replay` per (suite, k).
- Statistics as EXP-0002: Spatial + Object pooled per seed, ratio to the student's own teacher (mean over seeds / mean over seeds), bootstrap 95% CI over seeds, per-seed values reported.
- Student widths (d) and head parameters for `track` vs EXP-0002.

## 5. Pre-registered criteria

All on Spatial + Object pooled (mean of the two suites per seed); "teacher" = the H40 teacher's own success (EXP-0002: 68.9%).

| | Supported if | Refuted if |
|---|---|---|
| **H1, primary (100k student, k = 20)** | student(`track`) ≥ 50% of its teacher's success **and** student(`track`) − student(EXP-0002) ≥ 15 pp | student(`track`) − student(EXP-0002) < 5 pp |

Reported, not gating:
- the same comparison at k = 1, 4, 10 (100k) and at all k for 10k (a structure that helps at k = 20 should not hurt at k ≤ 4);
- `replay` at each k, and student(`track`) − `replay` (what the residual adds over the plan alone);
- reading of `replay` at k = 20: if `replay`(k = 20) < 50% of `replay`(k = 1), the plan itself is too stale to execute open loop at that age; otherwise an EXP-0002-style failure at k = 20 lies with the student.

**Why 100k and k = 20 (author's choice):** it targets the remaining collapse directly (EXP-0002: 11.6%, 0.17 of the teacher). The size question is reported (10k) but not gating.

## 6. Threats to validity

- **Two mechanisms in one arm (author's choice).** If H1 holds, the window and the residual are not separated; a follow-up can take them apart.
- **The window is also a compression.** z shrinks from 280-d to 56-d, so at a fixed head budget the student's own width grows (EXP-0002: d 16 / 76 at 10k / 100k with 280-d z; EXP-0001: d 20 / 84 with 56-d z). Overlaps with queue #6 (latent compression), most of all at 10k. Report widths.
- **The residual bound is a design choice** (0.5, arm dimensions only), taken from one source on LIBERO, not tuned. A failure could be the bound; reported with the residual magnitudes the student actually uses (fraction of steps at the bound).
- **The bound is uneven in normalized units** (observed in the debug run, before any result; not changed): 0.5 raw is ~1.0–1.4 action std on position but ~7–13 std on rotation, whose actions are small. In effect the bound constrains position only.
- **`replay` before the first plan** acts on the teacher's own fresh output for the first k steps; this favours `replay` slightly at k = 20 (20 fresh steps per episode).

## 7. Plan and cost

1. Code: `track` option, aligned window (training and evaluation), `replay` mode; tests (§3).
2. Debug runs on the debug partitions: one 10k `track` student for 200 steps on EXP-0002 seed-0 latents; a 2-episode eval at k = 20; a 2-episode `replay` at k = 20.
3. 6 `replay` evals (one job per teacher, all 4 k; CPU).
4. 12 students (2 suites × 2 sizes × 3 seeds, CPU, ~20–40 min each), 48 evals (one per run and k, ~30–45 min each), submitted by the cluster-side submitter.

No GPU time needed (teachers and latents reused).

## 8. Results

_Not started._

## Change log

- 2026-10-30: initial draft. Scope (one combined arm), primary location (100k, k = 20) and no-plan handling (act unbounded) chosen by the author.
- 2026-10-30: criteria approved by the author before any run; code, tests (59 pass), manifests (`scripts/exp0003_manifests.py`) and the analysis script (`scripts/exp0003_summarize.py`) written before any run.
- 2026-10-30 (debug runs, before any full run): 10k `track` student trains (d = 20, 9,036 params, z 56-d; val L1 0.123 after 200 steps vs 0.257 for the EXP-0002 debug student); `replay` and student evals run at k = 20. Added a test that a zero-residual student executes exactly what `replay` executes once a plan has arrived. Noted the uneven bound in §6. No criterion changed.
