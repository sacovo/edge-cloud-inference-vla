# Experiment queue

One question per experiment, each with its own pre-registration and result; the results are combined at the end (author, 2026-10-30). New ideas go here, not into a running experiment. Order matters: each experiment fixes something the next one builds on.

Background for all of them: EXP-0001 showed that a 100k fast policy reaches 84–95% of a 10M teacher's success when it receives the teacher's 8-step plan every ≤ 4 steps (0.2 s) with the same delay, and collapses at k ≥ 10; a 10k policy reaches 48% of the teacher (H4 bar: 50%). LIBERO does not need scene state (A-test fails), so state questions need other tasks.

| # | Experiment | Question | Status |
|---|---|---|---|
| 1 | EXP-0001 | Phase A on LIBERO: does a blind policy need scene state (A-test)? Can a small policy use a big policy's plan (H4)? | closed (Phase A) |
| 2 | EXP-0002 plan horizon | Is the collapse at k ≥ 10 caused by the plan running out (8 steps) rather than by the small policy? | done: H1 supported (k = 10: +21.8 pp at 100k); k = 20 still collapses |
| 3 | Tracking structure | Do an age-aligned plan and a residual action head (bounded, per step) bring a 10k policy to ≥ 50% of the teacher? | queued |
| 4 | Late arrivals | Under jittered, out-of-order delay: stale plan vs rewind-and-replay vs a small JEPA-style latent predictor that rolls z forward | queued |
| 5 | Innovation trust | Under an injected grasp loss: trust in z set by its age vs by the force/torque innovation (and the latent innovation); then innovation-triggered re-query with a max-interval fallback | queued |
| 6 | Latent compression | Does compressing z to a few dimensions stop it from overwhelming a 10k student and raise its success further? | queued (added by the author 2026-10-30) |

**#3 notes (from EXP-0002, 2026-10-30):** include a reference arm where the teacher executes its own delayed plan (chunk step j = age, no student) at k = 10 and 20. It separates "the plan is stale" from "the student cannot use it": EXP-0002's post-hoc check found the teacher's plan error grows only gradually with age on demos (0.100 at j 10–19, 0.122 at j 20–39), while student success falls from 60.5% to 11.6%. The test point with the largest gap is k = 20.

**#6 notes (to design later):** in EXP-0001 the 10k student did well with the 56-d chunk latent (48% of the teacher) and failed with the 768-d hidden latent (≤ 20%); EXP-0002's 280-d chunk raises the same risk. At a fixed head budget, a wide z costs the student its own width. Options: a learned bottleneck on the slow side (projection or small autoencoder of the teacher's chunk or hidden state to d_z ∈ {2, 4, 8, 16, 32}, trained with the student or distilled), or a compact parametric plan (few B-spline/DCT coefficients, Theme 7). One variable: d_z, at fixed student size (10k), against the uncompressed z. A small z also cuts the bytes sent per update (P-remote).

## Later, independent of the chain

- **VLM latents (was EXP-0001 A1b/A2):** FAST fine-tune of `pi05_base`, then the student setup with a real VLM latent instead of the teacher. Probe every candidate z for continuous and future state first (ALPS-style probe, Theme 8), including V-JEPA 2 as a slow-side encoder (H0).
- **State hypotheses (was EXP-0001 Phase B):** tasks where the needed state is unobservable from proprio and force/torque and changes during the episode (observability rule; object moved while not touching it, moving targets).
- **Infrastructure:** backbone-only latency on an H200, the Jetson Orin NX/Nano benchmark, full-vision eval timing.

## Sources for the mechanisms

- Plan + feedback, delay handling, innovation gating, event triggers: `literature/themes/07-learned-filtering-in-policies.md` (Sec. 1 and 3).
- Latent predictors and JEPA: `literature/themes/08-predictive-latents-jepa.md` (Sec. 1 and 3).
- The retired all-in-one draft ("A1-teacher v2", 9 variants) is in git history (commit 43cf56e); the experiments above split it into single questions.
