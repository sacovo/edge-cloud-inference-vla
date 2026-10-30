# EXP-0002: Is the collapse at slow update rates the plan running out?

- **Status:** done (pre-registered 2026-10-30; all runs and evals 2026-10-30). H1 supported.
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

### H1 (primary, 100k, k = 10): **supported** (2026-10-30)

Spatial + Object pooled, 3 seeds, 500 episodes per seed, suite and k: the H40 student reaches **60.5%** vs 38.7% for the H8 student (EXP-0001), **+21.8 pp** (per seed +24.2 / +13.5 / +27.8; bootstrap 95% CI over seeds 13.5–27.8), and **0.88** of its own teacher (per seed 0.80 / 0.89 / 0.95). Both parts of the criterion (≥ 50% of own teacher; ≥ 15 pp over H8) hold on the means, as pre-registered. One of three seeds (s1) is below 15 pp, so the CI's lower end is under the bar; with 3 seeds the bootstrap CI is essentially the seed range.

Summary: `results/exp0002_summary.json` (from `scripts/exp0002_summarize.py`, committed before any run); inputs `results/teachers_h40.json`, `results/students_h40.json` (`scripts/a1_summarize.py`). H8 numbers are EXP-0001's (`a1_teacher_summary.json`, chunk latent; teachers from `a1_grid_blind_summary.json`).

### Reported, not gating

Success %, mean of 3 seeds (H8 → H40):

| Suite | Teacher | 10k k=1 / 4 / 10 / 20 | 100k k=1 / 4 / 10 / 20 |
|---|---|---|---|
| Spatial | 75.9 → 56.7 | 31.1 → 17.8 / 33.4 → 34.5 / 7.6 → 9.9 / 1.0 → 4.6 | 64.1 → 70.3 / 68.7 → 65.5 / 27.4 → 59.7 / 5.9 → 15.3 |
| Object | 97.3 → 81.1 | 51.7 → 25.8 / 55.0 → 52.6 / 2.4 → 1.9 / 0.1 → 0.2 | 92.3 → 85.0 / 94.7 → 82.8 / 49.9 → 61.3 / 5.1 → 8.0 |
| Pooled | 86.6 → 68.9 | 41.4 → 21.8 / 44.2 → 43.6 / 5.0 → 5.9 / 0.5 → 2.4 | 78.2 → 77.6 / 81.7 → 74.1 / 38.7 → 60.5 / 5.5 → 11.6 |

Pooled difference H40 − H8 (pp, per seed) and ratio to own teacher (H8 → H40):

| Size | k | diff (seeds) | ratio |
|---|---|---|---|
| 10k | 1 | −19.6 (−16.8 / −13.3 / −28.7) | 0.48 → 0.32 |
| 10k | 4 | −0.6 (−19.8 / +8.7 / +9.2) | 0.51 → 0.63 |
| 10k | 10 | +0.9 (−1.2 / +6.1 / −2.1) | 0.06 → 0.09 |
| 10k | 20 | +1.9 (+0.6 / +2.8 / +2.2) | 0.01 → 0.03 |
| 100k | 1 | −0.6 (+2.7 / +1.1 / −5.5) | 0.90 → 1.13 |
| 100k | 4 | −7.6 (−4.2 / −5.9 / −12.6) | 0.94 → 1.08 |
| 100k | 10 | **+21.8** (+24.2 / +13.5 / +27.8) | 0.45 → **0.88** |
| 100k | 20 | +6.1 (+9.5 / +4.8 / +4.1) | 0.06 → 0.17 |

- **The H40 teacher is weaker** (pooled 68.9% vs 86.6%; every seed lower). Predicting 40 steps is a harder target, and at the 10M budget its width drops (d 768 → 704; 10.74M → 9.19M head params). Not separated here.
- **k = 20 still collapses** (100k: 11.6%, 0.17 of the teacher) although the 40-step plan covers its oldest age (39). The plan running out explains most of the k = 10 drop, not the k = 20 one: a plan 20–39 steps old is not usable by this student even when it still covers the present (reader-inferred; open-loop drift over 1–2 s is the obvious candidate, not tested).
- **100k students match or exceed their own (weaker) teacher at k ≤ 4** (ratio 1.08–1.13), and lose 7.6 pp against H8 at k = 4 (mostly Object, −11.9).
- **The 10k student gets worse with the 280-d z** (k = 1: −19.6 pp, every seed lower; no gain at k = 10). Widths: 10k d 20 → 16, 100k d 84 → 76. Consistent with the §5/§6 risk that a wide z crowds out a 10k head (queue #6), but not separated from the weaker teacher.

### Post-hoc (author's question, not gating): does the plan go stale with age?

Teacher L1 per chunk step j on held-out demos (teacher-forced; same split and units as train.py's val L1, which j = 0 reproduces), pooled Spatial + Object, 3 seeds; at period k and delay k the student uses steps j = k .. 2k − 1. Persistence = predicting a_{t+j} = a_t. Script `scripts/exp0002_chunk_error.py`, data `results/chunk_error.json`.

| Steps used | H40 teacher L1 | persistence L1 | 100k student success |
|---|---|---|---|
| j = 0 | 0.066 | 0 | |
| k = 4: j 4–7 | 0.088 (H8: 0.073) | 0.117 | 74.1% (H8: 81.7%) |
| k = 10: j 10–19 | 0.100 | 0.253 | 60.5% |
| k = 20: j 20–39 | 0.122 | 0.371 | 11.6% |

- The plan degrades with age (×1.85 from j = 0 to j 20–39; position 0.083 → 0.170, gripper 0.154 → 0.247, rotation ~flat), but gradually: +22% from the k = 10 to the k = 20 window, against a drop in success from 60.5% to 11.6%. Old steps still beat persistence 3×.
- On demos, staleness alone is too gradual to explain the cliff. Not tested: closed-loop divergence (teacher-forcing makes this a lower bound) and a precision threshold (grasping may tolerate ~0.10 but not ~0.12). The separating test is the teacher executing its own delayed plan at k = 20 (no student).

## Change log

- 2026-10-30: initial design; criteria approved by the author before any run.
- 2026-10-30 (after the run, no criterion changed): debug runs and teachers ran as planned; teachers on `h200` (15–19 min each), students on CPU (16–38 min), evals ~30–45 min each. Students were submitted with a Slurm dependency on the latent array; evals via the cluster-side submitter. Deviation from §7's cost estimate only (evals faster than 1.5 h). No failed or crashed runs.
- 2026-10-30 (after the result): added the post-hoc per-step plan-error analysis (author's question); not gating, no criterion changed.
