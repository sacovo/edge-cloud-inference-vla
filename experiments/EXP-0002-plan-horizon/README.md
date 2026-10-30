# EXP-0002: Is the collapse at slow update rates the plan running out?

- **Status:** running (pre-registered 2026-10-30; debug runs passed, full runs started 2026-10-30)
- **Platform:** LIBERO-Spatial and LIBERO-Object, blind fast policies, simulated delay (same setup as EXP-0001 A1-teacher)
- **Related:** EXP-0001 §10 (H4 results), `experiments/QUEUE.md` (#2); literature Theme 7 Sec. 1 (plans that outlive the delay; DTC age-indexed plans).

The hypothesis, conditions and criteria below are fixed **before** any run. Changes after the first result go in the Change log with a reason.

## 1. Question

In EXP-0001, a 100k student receiving the 10M teacher's 8-step action chunk reached 84–95% of the teacher's success at update period k ≤ 4, but fell to 27–50% at k = 10 and 5–11% at k = 20. With update period k and delivery delay k, the plan in use is between k and 2k − 1 steps old. An 8-step plan has therefore run out before it arrives whenever k ≥ 8. Is the collapse caused by that, or by the small policy's inability to work from an old plan?

## 2. Hypothesis

- **H1 (horizon).** If the teacher's plan covers the whole delay, the student keeps most of the teacher's success at slow update rates. The EXP-0001 collapse at k ≥ 10 is mainly the plan running out.
- **H0 (null).** A longer plan does not help: an old plan is not useful to the student even if it still covers the present.

## 3. Design

One change against EXP-0001's A1-teacher: the teacher's plan horizon.

| | EXP-0001 (baseline, reused) | EXP-0002 (new) |
|---|---|---|
| Teacher | 10M blind `task-id` GRU, chunk H = 8 | same, chunk **H = 40** (2 s), so a plan delivered at k = 20 still covers its oldest age (39 steps) |
| z | teacher's predicted chunk, flattened (8 × 7 = 56) | same (40 × 7 = 280) |
| Student | blind, 10k and 100k head, task token + one z token with z's age; own 8-step chunk, first action executed | identical |
| Training delay | k ∈ {1, 4, 10, 20} per episode, delay k − 1..k + 1 | identical |
| Evaluation | 50 fixed LIBERO initial states × 10 tasks; z read every k steps, delivered exactly k steps late; k ∈ {1, 4, 10, 20} | identical |
| Seeds | 3 (student seed s uses teacher seed s) | 3 |

- **Baseline reuse.** The H = 8 numbers are EXP-0001's A1-teacher results (`EXP-0001-latent-taskid-vs-state/results/a1_teacher_summary.json`), produced by the same code paths, data (`prepared_v2`), demo filtering, validation split and evaluation protocol. They are not re-run.
- **Only code change:** the chunk length becomes a model setting (default 8, so EXP-0001 behaviour is unchanged; covered by tests).
- **Suites:** Spatial + Object only (the H4 pooling). Goal is not run.

## 4. Measurements

- Success rate per (suite, teacher horizon, student size, k), mean and range over 3 seeds.
- The H = 40 teacher's own success (evaluated like every policy). A long-horizon teacher may itself be better or worse than the H = 8 one, so the student criteria are relative to **its own** teacher.
- Statistics: mean over seeds with the per-seed values reported; 95% CI from a bootstrap over seeds.

## 5. Pre-registered criteria

All on Spatial + Object pooled (mean of the two suites per seed).

| | Supported if | Refuted if |
|---|---|---|
| **H1, primary (100k student, k = 10)** | student(H40) ≥ 50% of its teacher's success **and** student(H40) − student(H8) ≥ 15 pp | student(H40) − student(H8) < 5 pp |

Reported, not gating:
- the same comparison at k = 20 (100k), and at k = 10 and 20 for the 10k student;
- k = 1 and 4 for both sizes (a longer plan should not hurt when updates are fast);
- the H = 40 teacher's success vs the H = 8 teacher's.

**Why the primary criterion is at 100k:** the 10k student already falls short of the teacher at k = 1 (EXP-0001 H4: 48%), so a horizon effect would be hard to separate from its general weakness. Also, the 280-d z takes more of a 10k head's budget than the 56-d z did (reader-inferred risk, reported below).

## 6. Threats to validity

- **The teacher changes.** Predicting 40 steps instead of 8 changes the teacher's training target, so the teacher (and its plans) differ beyond the horizon. Mitigated by judging students relative to their own teacher, and by reporting both teachers.
- **Input size at fixed head size.** A 280-d z costs more parameters in the student's z encoder than a 56-d z; at a fixed head-size target the rest of the network gets narrower. Report the student widths (d) for both horizons.
- **Baseline from an earlier commit.** EXP-0001's numbers come from commit-tagged runs with the same code paths; the only code change here is the configurable chunk length, tested to leave H = 8 unchanged.

## 7. Plan and cost

1. Code: configurable chunk length; tests (H = 8 unchanged; H = 40 shapes; stepwise = sequence).
2. Debug runs on the debug partitions (one teacher for 200 steps; latents; one student for 200 steps; a 2-episode eval).
3. 6 teachers (2 suites × 3 seeds, GPU, ~20–30 min each), their 6 evals; teacher latents (CPU, < 1 min each).
4. 12 students (2 suites × 2 sizes × 3 seeds, CPU or GPU), 48 eval jobs (one per run and k, ~1.5 h each on CPU).

## 8. Results

_Not started._

## Change log

- 2026-10-30: initial design; criteria approved by the author before any run.
