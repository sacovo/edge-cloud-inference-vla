# EXP-0003: Does a tracking structure let a small policy use an old plan?

- **Status:** done (pre-registered 2026-10-30; runs 2026-10-30/31; H1 inconclusive)
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

### H1 (primary, 100k, k = 20): **neither supported nor refuted (inconclusive)** (2026-10-31)

Spatial + Object pooled, 3 seeds, 500 episodes per seed, suite and k: the `track` student reaches **22.4%** vs 11.6% for the EXP-0002 student, **+10.7 pp** (per seed +9.6 / +10.8 / +11.8; bootstrap 95% CI over seeds 9.6–11.8), and **0.32** of its teacher (per seed 0.36 / 0.29 / 0.33). Support needed ≥ 15 pp and ≥ 0.50; refutation needed < 5 pp. Both parts miss support, and the gain is above the refutation bar on every seed, so the pre-registered verdict is inconclusive. The gain is consistent across seeds; it is too small, not noisy.

Summary: `results/exp0003_summary.json` (from `scripts/exp0003_summarize.py`, committed before any run); inputs `results/students_track.json` and `results/replay.json` (`scripts/a1_summarize.py`, the latter with `--eval-dir eval_replay`). Baselines are EXP-0002's `students_h40.json` and `teachers_h40.json` (teacher pooled 68.9%).

### Reference arm `replay` (the teacher executes its own delayed plan)

Pooled, mean of 3 seeds: **76.8 / 76.3 / 70.8 / 40.2%** at k = 1 / 4 / 10 / 20. Pre-registered reading: replay(k = 20) / replay(k = 1) = **0.52**, just above the 0.5 line, so by the rule the plan is not too stale to execute open loop and the k = 20 shortfall lies with the student. The margin is small (0.02), and at k = 20 the plan alone still loses half its success; both halves of H0 contribute.

- **The student is worse than the plan it receives at long ages:** `track` − `replay` = −17.9 pp at k = 20 and −8.1 pp at k = 10 (100k); −15.6 / −4.5 pp at 10k. At k ≤ 4 the two are within ±4 pp. The bounded residual, trained on demos with the training delay, does not correct a stale plan in closed loop; it makes it worse.
- **`replay` at k = 1 beats the teacher's own closed-loop run** (76.8% vs 68.9%; every seed, both suites: Spatial 64.1 vs 56.7, Object 89.6 vs 81.1). At k = 1 replay executes step 1 of the plan made one step earlier instead of step 0 of a fresh one. Not explained here (reader-inferred candidates: the one-step offset matches the sensor-to-action lag, or step 1 is smoother); it does not enter any criterion.

### Reported, not gating

Success %, mean of 3 seeds (EXP-0002 → `track`); `replay` for reference:

| Suite | replay k=1 / 4 / 10 / 20 | 10k k=1 / 4 / 10 / 20 | 100k k=1 / 4 / 10 / 20 |
|---|---|---|---|
| Spatial | 64.1 / 66.3 / 64.5 / 47.5 | 17.8 → 60.9 / 34.5 → 66.0 / 9.9 → 62.1 / 4.6 → 28.7 | 70.3 → 63.7 / 65.5 → 67.5 / 59.7 → 53.5 / 15.3 → 29.4 |
| Object | 89.6 / 86.3 / 77.1 / 32.9 | 25.8 → 85.4 / 52.6 → 93.8 / 1.9 → 70.6 / 0.2 → 20.6 | 85.0 → 89.7 / 82.8 → 90.6 / 61.3 → 72.0 / 8.0 → 15.3 |
| Pooled | 76.8 / 76.3 / 70.8 / 40.2 | 21.8 → 73.1 / 43.6 → 79.9 / 5.9 → 66.3 / 2.4 → 24.6 | 77.6 → 76.7 / 74.1 → 79.0 / 60.5 → 62.8 / 11.6 → 22.4 |

Pooled difference `track` − EXP-0002 (pp, per seed), ratio to the teacher (EXP-0002 → `track`), and `track` − `replay`:

| Size | k | diff (seeds) | ratio | − replay (pp) |
|---|---|---|---|---|
| 10k | 1 | +51.3 (+58.5 / +54.4 / +41.1) | 0.32 → 1.06 | −3.7 |
| 10k | 4 | +36.3 (+56.0 / +35.2 / +17.8) | 0.63 → 1.16 | +3.6 |
| 10k | 10 | +60.4 (+64.8 / +60.6 / +55.8) | 0.09 → 0.96 | −4.5 |
| 10k | 20 | +22.2 (+22.3 / +16.5 / +27.9) | 0.03 → 0.36 | −15.6 |
| 100k | 1 | −1.0 (−0.1 / −9.1 / +6.3) | 1.13 → 1.11 | −0.2 |
| 100k | 4 | +4.9 (+3.0 / +6.6 / +5.1) | 1.08 → 1.15 | +2.7 |
| 100k | 10 | +2.3 (+5.2 / +3.2 / −1.6) | 0.88 → 0.91 | −8.1 |
| 100k | 20 | **+10.7** (+9.6 / +10.8 / +11.8) | 0.17 → **0.32** | −17.9 |

- **At 100k and k ≤ 10 the structure neither helps nor hurts** (−1.0 to +4.9 pp), as §5 asked.
- **The 10k student changes most:** from 5.9% to 66.3% at k = 10 (0.96 of the teacher) and from 21.8% to 73.1% at k = 1. Not attributable to tracking alone: the window also shrinks z from 280-d to 56-d (§6; widths d 16 → 20 at 10k, 76 → 84 at 100k; head params 9,036 / 102,284), and at k ≤ 10 the 10k student stays within 4.5 pp of `replay`, i.e. it mostly executes the plan. What this shows is that a 9k-parameter policy can execute a 10M teacher's delayed plan about as well as the teacher can, up to k = 10.
- Per suite, the k = 20 gain at 100k is larger on Spatial (+14.1 pp) than on Object (+7.3 pp); Object's `replay` also falls further (89.6 → 32.9%).

### Residual use (§6, not gating)

Executed first action on held-out demos (2 per task, train.py's val split), teacher-forced, evaluation delay exactly k, steps after the first plan only; mean of 6 runs per size (2 suites × 3 seeds). Bound = 0.5 raw on each arm dimension. Script `scripts/exp0003_residual_bound.py` (written after the results; run locally on a copy of the cluster data), data `results/residual_bound.json`.

| Size | k | arm (dim, step) ≥ 95% of bound | pos ≥ 50% of bound | mean / p95 \|r\| pos (raw) | mean \|r\| rot (raw) | mean \|r\| grip (raw) | L1 student / plan alone |
|---|---|---|---|---|---|---|---|
| 10k | 1 | 0 | 0 | 0.002 / 0.004 | 0.000 | 0.001 | 0.074 / 0.074 |
| 10k | 10 | 0 | 0.09% | 0.005 / 0.024 | 0.001 | 0.002 | 0.096 / 0.097 |
| 10k | 20 | 0 | 0.41% | 0.012 / 0.076 | 0.002 | 0.003 | 0.113 / 0.116 |
| 100k | 1 | 0 | 0 | 0.004 / 0.012 | 0.001 | 0.001 | 0.073 / 0.074 |
| 100k | 10 | 0 | 0.43% | 0.010 / 0.039 | 0.002 | 0.001 | 0.094 / 0.097 |
| 100k | 20 | 0.07% of steps (any arm dim) | 1.5% | 0.025 / 0.138 | 0.005 | 0.003 | 0.108 / 0.116 |

- **The bound is not what limits the student.** At 100k, k = 20, a residual reaches 95% of the bound on 0.07% of steps (0.02% of position entries); the mean position residual is 0.025 raw, 5% of the bound. A failure from "the bound is too tight" (§6) is ruled out on demos.
- **The students barely correct the plan.** The residual grows with age (100k: 0.004 → 0.025 raw from k = 1 to 20) and lowers teacher-forced L1 by 7% at k = 20 (0.116 → 0.108), but the gripper residual stays ~0.003 against actions of ±1: the student never moves the gripper timing that degrades most with age (EXP-0002 post-hoc).
- **So the closed-loop gap to `replay` (−17.9 pp at k = 20) is not explained on demos**, where student ≈ plan. Reader-inferred candidates, not tested: residuals grow off the demo distribution in closed loop; and the first k steps, where the student acts alone (its own blind output, no base) while `replay` uses the teacher's fresh output (§6 noted this favours `replay`; at k = 20 these are the first 20 of each episode's steps).

## Change log

- 2026-10-30: initial draft. Scope (one combined arm), primary location (100k, k = 20) and no-plan handling (act unbounded) chosen by the author.
- 2026-10-30: criteria approved by the author before any run; code, tests (59 pass), manifests (`scripts/exp0003_manifests.py`) and the analysis script (`scripts/exp0003_summarize.py`) written before any run.
- 2026-10-30 (debug runs, before any full run): 10k `track` student trains (d = 20, 9,036 params, z 56-d; val L1 0.123 after 200 steps vs 0.257 for the EXP-0002 debug student); `replay` and student evals run at k = 20. Added a test that a zero-residual student executes exactly what `replay` executes once a plan has arrived. Noted the uneven bound in §6. No criterion changed.
- 2026-10-30 (ops, no design change): all jobs queued behind a full CPU cluster (calc-c-001..004 reserved for another group, 006/007 full). Time limits lowered to help backfill: students 3 h, replay 5 h, submitter 10 h (EXP-0002 needed ≤ 38 min per student, 30–45 min per eval k). Author chose to keep waiting rather than run CPU jobs on the GPU node.
- 2026-10-31 (results): all 6 replay jobs, 12 students and 48 evals completed (no failures). §8 filled from the pre-registered summary script; H1 inconclusive. The §6 residual-bound check was added as `scripts/exp0003_residual_bound.py` after the results (not gating); first queued on cpu-debug (job 122419, never started: nodes full or reserved), cancelled and run locally on a copy of the cluster data instead. No criterion changed.
