# EXP-0004: Does the start-up phase explain why the `track` student falls short of the plan it receives?

- **Status:** done (pre-registered 2026-10-31; runs 2026-11-01; H1 supported)
- **Platform:** LIBERO-Spatial and LIBERO-Object, blind fast policies, simulated delay (same setup as EXP-0003)
- **Related:** EXP-0003 §8 (`replay` reference arm, residual use on demos), `experiments/QUEUE.md`.

The hypothesis, conditions and criteria below are fixed **before** any run. Changes after the first result go in the Change log with a reason.

## 1. Question

In EXP-0003 the 100k `track` student reached 22.4% at k = 20, against 40.2% for the teacher executing its own delayed plan (`replay`): a gap of **17.9 pp** (8.1 pp at k = 10). On held-out demos the same students barely correct the plan (mean position residual 0.025 raw against a bound of 0.5; teacher-forced L1 within 7% of the plan alone), so on demo states the student ≈ the plan. The gap appears only in closed loop.

The two arms differ in one place by construction: **before the first plan arrives** (the first k steps of each episode), the student acts on its own blind output (no plan, residual not bounded), while `replay` executes the 10M teacher's fresh output (EXP-0003 §6 noted this favours `replay`). At k = 20 that is the first 20 steps of every episode.

Does holding the robot still until the first plan arrives, in both arms, close the gap?

## 2. Hypotheses

- **H1 (start-up).** The gap comes from the start-up phase: the student's own blind actions in the first k steps put the robot into states from which the student, following the plan, does not recover. With a hold start in both arms, the gap closes.
- **H0 (null).** The gap persists with a hold start: the student degrades the plan later in the episode (candidate: residuals grow off the demo distribution in closed loop; measured, §4).

## 3. Design

One change against EXP-0003: how the episode starts. No training; the EXP-0003 students and the EXP-0002 H40 teachers are reused.

| | EXP-0003 (no hold) | EXP-0004 `hold` |
|---|---|---|
| Steps before the first plan (t < k) | student: own blind output (base 0, residual unbounded); `replay`: teacher's fresh output | **robot holds still** until the first plan arrives, in both arms |
| First plan | arrives at t = k, used at its age (k .. 2k − 1) | the plan made from the first observation, **treated as fresh**: used at ages 0 .. k − 1 (author's choice; nothing moved during the hold, so it still matches the current state) |
| Later plans | query every k steps, delay exactly k | identical (same schedule from the first acting step on) |
| Step budget | 220 (Spatial) / 280 (Object) acting steps | identical; **hold steps are not counted** (author's choice) |

- **Implementation.** After LIBERO's 10 settle steps the scene is static, so holding k steps with the hold not counted is equivalent to the slow model having answered the first query before the robot starts: at acting step t < k the student (and `replay`) gets the plan from the first observation at age t; from t = k on, the EXP-0003 schedule. In code: an eval switch `--hold-start` that sets the schedule's source to step 0 where it was "no plan yet" (src = −1 → 0). The robot does not physically wait in the sim (threat, §6).
- **Arms** (each at k ∈ {10, 20}; k = 1 and 4 change by ≤ 4 steps per episode and are not run):
  - `track-hold`: the 12 EXP-0003 students (2 suites × 10k/100k × 3 seeds) with `--hold-start`;
  - `replay-hold`: the 6 EXP-0002 H40 teachers, `--replay --hold-start`;
  - `track-nohold`: the same 12 students without the hold, rerun only to log closed-loop residuals (below). Its success rates must reproduce EXP-0003's (same code path, same seed); any difference is reported.
- **Closed-loop residual logging (author approved the eval change).** For `track` students, the eval also records per step the bounded residual of the executed action (|r_b| / bound on the arm dimensions, |r| on the gripper, raw units) and summarises per episode (mean, p95, fraction ≥ 95% / ≥ 50% of the bound, over the whole episode and per 40-step window). Logging only; actions unchanged (test).
- **Evaluation:** as EXP-0003 (50 initial states × 10 tasks per suite, eval seed 0).
- **Code changes:** `--hold-start` in `evaluate.py` (Runner and ReplayRunner), residual logging in the Runner for `track` models. Tests: hold-start schedule (src −1 → 0, ages 0..k−1 for the first plan, unchanged afterwards); actions with logging equal actions without; `--hold-start` off reproduces the EXP-0003 schedule.

## 4. Measurements

- Success per (suite, arm, size, k), mean and range over 3 seeds; pooled Spatial + Object per seed; bootstrap 95% CI over seeds (as EXP-0003, `scripts/exp0003_summarize.py` conventions).
- **Gap** = `replay` − `track` (pp, pooled), with hold and without (the latter from the `track-nohold` rerun and EXP-0003's `replay`).
- Closed-loop residuals per (arm, size, k), split by successful vs failed episodes and by episode time; compared with the teacher-forced values from EXP-0003 (`results/residual_bound.json`).

## 5. Pre-registered criteria

On Spatial + Object pooled, 3 seeds, 100k students, **k = 20**. gap_nohold is measured in this experiment (`replay` from EXP-0003, `track-nohold` rerun; expected 17.9 pp).

| | Supported if | Refuted if |
|---|---|---|
| **H1, primary (100k, k = 20)** | gap_hold ≤ ⅓ · gap_nohold (≈ 6 pp) | gap_hold ≥ ⅔ · gap_nohold (≈ 12 pp) |

In between: partial (the start-up explains part of the gap).

Reported, not gating:
- the same at k = 10 (gap_nohold 8.1 pp) and for 10k (15.6 / 4.5 pp at k = 20 / 10);
- `track-hold` against EXP-0003's bars (≥ 50% of the teacher, ≥ +15 pp over EXP-0002's 11.6%) — a hold start could pass them, but EXP-0003's verdict stays as recorded;
- `replay-hold` − `replay` (how much the teacher's fresh start-up output helped `replay`);
- **closed-loop residual reading (H0 candidate):** residuals "grow in closed loop" if, at 100k k = 20 with hold, the mean position residual is ≥ 2× the teacher-forced value (≥ 0.05 raw) or ≥ 1% of steps reach 95% of the bound; and whether failed episodes carry larger residuals than successful ones.

## 6. Threats to validity

- **Hold is implemented as an equivalent schedule, not a physical wait.** Valid if the scene is static after the 10 settle steps and the robot does not drift under zero-delta commands; objects that are still settling or a slowly drifting arm would make a real hold differ. Check on the debug run: hold the robot for 20 no-op steps after settling and report the largest end-effector and object displacement.
- **The first plan at age 0** is a fresh plan the students saw in training (k = 1 draws ages 0–2), but never at the start of an episode with a full k-step window ahead. Not separated.
- **`replay-hold` changes too:** its first k steps become the teacher's open-loop plan instead of its closed-loop output. The gap compares two changed arms; both are reported.
- **The rerun reproduces EXP-0003 only if the eval is deterministic.** Residual logging adds no randomness; a mismatch would point to nondeterminism in the sim or threads, reported.

## 7. Plan and cost

1. Code: `--hold-start`, residual logging, tests (§3).
2. Debug (cpu-debug): one 100k student, 2 episodes at k = 20 with and without hold; one `replay-hold`; the 20-step static-hold drift check.
3. Full: 24 `track-hold` + 24 `track-nohold` evals (one per run and k) and 6 `replay-hold` jobs (one per teacher, both k), CPU, ~30–45 min each, via the cluster-side submitter. No training, no GPU. Analysis script written before any full run.

## 8. Results

### H1 (primary, 100k, k = 20): **supported** (2026-11-02)

Spatial + Object pooled, 3 seeds, 500 episodes per seed, suite and arm. Without hold (rerun): `replay` 40.2%, `track` 22.4%, gap **17.9 pp** (per seed 18.1 / 17.1 / 18.4). With hold: `replay-hold` **65.1%**, `track-hold` **61.1%**, gap **4.0 pp** (per seed 6.0 / 0.2 / 5.7; bootstrap 95% CI over seeds 0.2–6.0), **0.22** of the no-hold gap. The bar was ≤ ⅓ (≈ 6.0 pp) on the means; it holds. Seed s0 (6.0 vs its own 18.1) is right at ⅓.

Summary: `results/exp0004_summary.json` (from `scripts/exp0004_summarize.py`, committed before any run); inputs `results/students_hold.json`, `results/students_log.json`, `results/replay_hold.json` (`scripts/a1_summarize.py` with `--eval-dir eval_hold / eval_log / eval_replay_hold`).

**Reproduction check (§6):** the no-hold rerun matches EXP-0003 exactly on Object; on Spatial 11 of 24 (run, k) cells differ by 0.2–1.2 pp (max |diff| 0.012). The 2-episode debug run matched exactly. So the eval is not fully deterministic on Spatial (cause not identified: sim or thread scheduling); the effect is small against the effects measured here. The no-hold numbers in this section are the rerun's.

### The hold helps both arms, and most at k = 20

Success %, pooled, mean of 3 seeds (no hold → hold):

| Size | k | `track` | `replay` | gap (pp) | gap ratio | `track-hold` / teacher |
|---|---|---|---|---|---|---|
| 10k | 10 | 66.2 → 74.8 | 70.8 → 74.6 | +4.6 → −0.2 | −0.04 | 1.09 |
| 10k | 20 | 24.4 → 55.5 | 40.2 → 65.1 | +15.8 → +9.5 (seeds 0.9 / 23.7 / 4.0) | 0.60 | 0.81 |
| 100k | 10 | 62.7 → 74.2 | 70.8 → 74.6 | +8.1 → +0.4 | 0.05 | 1.08 |
| 100k | 20 | 22.4 → **61.1** | 40.2 → **65.1** | +17.9 → **+4.0** | **0.22** | **0.89** |

Per suite, hold (no hold): Spatial `track` 100k k = 20 52.1 (29.4), `replay` 54.7 (47.5); Object `track` 70.1 (15.3), `replay` 75.4 (32.9).

- **Most of the k = 20 collapse seen in EXP-0002 and EXP-0003 was the start-up phase, not plan staleness.** With a hold start the 100k student reaches 61.1% at k = 20 (0.89 of the teacher, +49.5 pp over EXP-0002's 11.6%), and the teacher replaying its own plan 65.1% (vs 76.8% at k = 1 without hold). A 1 s update period with 1 s delay is usable when the robot waits for the first plan.
- `track-hold` would pass EXP-0003's bars (≥ 0.50 of the teacher, ≥ +15 pp); EXP-0003's verdict stays as recorded (its protocol had no hold).
- At k = 10 the gap disappears for both sizes. At 10k, k = 20 a gap remains (9.5 pp, ratio 0.60 — "partial" if read by the primary rule), driven by seed s1 (23.7 pp; the other seeds 0.9 / 4.0).
- `replay` gains too (+24.8 pp at k = 20). Without hold its first k steps are the teacher's own fresh step-0 output; with hold they are steps 0..k−1 of its plan. Consistent with EXP-0003's unexplained observation that executing plan steps beats the teacher's fresh output (`replay` k = 1 76.8% vs the teacher's own 68.9%), but the mechanism is not tested here (reader-inferred). The student, which acts blind without hold, loses more (+38.7 pp with hold).

### Closed-loop residuals (H0 candidate, reported)

Executed residual, all episodes, step-weighted (raw units; bound 0.5). Teacher-forced reference (EXP-0003, 100k k = 20): 0.025.

| Size | k | mean \|r\| pos, no hold → hold | first 40 plan steps, no hold → hold | p95 pos, no hold → hold | gripper |
|---|---|---|---|---|---|
| 10k | 20 | 0.009 → 0.004 | 0.069 → 0.009 | 0.048 → 0.016 | 0.003 → 0.002 |
| 100k | 10 | 0.010 → 0.003 | 0.038 → 0.003 | 0.039 → 0.010 | 0.001 → 0.001 |
| 100k | 20 | 0.020 → **0.005** | **0.139** → 0.005 | 0.121 → 0.018 | 0.005 → 0.002 |

- **Pre-registered reading: residuals do not grow in closed loop** (100k k = 20 with hold: 0.005 raw, below the teacher-forced 0.025; no step at 95% of the bound).
- **Without hold, the residual spikes right after the first plan arrives** (100k k = 20: 0.139 raw over the first 40 plan steps, then 0.006–0.009): the student pulls hard against a plan that no longer fits where its own blind start took the robot. With hold the spike is gone.
- Without hold, successful episodes carry larger residuals than failed ones (100k k = 20: 0.032 vs 0.019), i.e. correcting harder after a bad start helps; with hold, successes and failures are alike (0.005 / 0.006).
- With hold, the student still adds nothing over the plan: residuals ~1% of the bound, and `track-hold` ≤ `replay-hold` at every cell but 10k k = 10.

## Change log

- 2026-10-31: initial draft. Idea (hold the robot until the first plan; investigate the closed-loop divergence before queue #4/#6), hold semantics (first plan treated as fresh), budget (hold steps not counted) and residual logging in the eval chosen by the author.
- 2026-10-31: criteria approved by the author before any run. Code: `--hold-start` and residual logging in `evaluate.py`; test (schedule src −1 → 0 with ages 0..k−1, unchanged after; logged residual = executed action − plan; zero-residual student = replay from step 0 with hold); 60 tests pass. Manifests (`scripts/exp0004_manifests.py`: 24 + 24 student evals, 6 replay jobs) and the analysis script (`scripts/exp0004_summarize.py`, dry-run on EXP-0003 data as stand-ins) written before any run.
- 2026-10-31 (debug, §6 drift check, before any eval): after the 10 settle steps, a 20-step no-op hold moves the end effector by ≤ 2.0 mm (median 0.4) and ≤ 0.23°, objects by ≤ 2.5 mm (one Spatial task; median 0), over 10 tasks × 5 initial states × 2 suites (`scripts/exp0004_hold_drift.py`, `results/hold_drift.json`). The schedule implementation of the hold is taken as valid. No criterion changed.
- 2026-11-01 (debug, before any full run): job 122623 on cpu-debug, Spatial task 0, 2 episodes at k = 20 for the 100k s0 student with and without hold and for `replay-hold` (outputs under `$ECVLA_SCRATCH/exp0004_debug`, separate from the real runs). All run; the no-hold rerun reproduces EXP-0003's episodes exactly (steps 73 / 83). Full runs: 122857 (`student_hold`, 24), 122858 (`student_log`, 24), 122859 (`replay_hold`, 6). No criterion changed.
- 2026-11-02 (results): all 54 jobs completed (no failures). §8 filled from the pre-registered summary script; H1 supported. The no-hold rerun reproduces EXP-0003 only up to 1.2 pp on Spatial (exactly on Object); reported in §8. No criterion changed.
