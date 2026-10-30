# Experiment queue

One question per experiment, each with its own pre-registration and result; the results are combined at the end (author, 2026-10-30). New ideas go here, not into a running experiment. Order matters: each experiment fixes something the next one builds on.

Background for all of them: EXP-0001 showed that a 100k fast policy reaches 84–95% of a 10M teacher's success when it receives the teacher's 8-step plan every ≤ 4 steps (0.2 s) with the same delay, and collapses at k ≥ 10; a 10k policy reaches 48% of the teacher (H4 bar: 50%). LIBERO does not need scene state (A-test fails), so state questions need other tasks.

| # | Experiment | Question | Status |
|---|---|---|---|
| 1 | EXP-0001 | Phase A on LIBERO: does a blind policy need scene state (A-test)? Can a small policy use a big policy's plan (H4)? | closed (Phase A) |
| 2 | EXP-0002 plan horizon | Is the collapse at k ≥ 10 caused by the plan running out (8 steps) rather than by the small policy? | designed |
| 3 | Tracking structure | Do an age-aligned plan and a residual action head (bounded, per step) bring a 10k policy to ≥ 50% of the teacher? | queued |
| 4 | Late arrivals | Under jittered, out-of-order delay: stale plan vs rewind-and-replay vs a small JEPA-style latent predictor that rolls z forward | queued |
| 5 | Innovation trust | Under an injected grasp loss: trust in z set by its age vs by the force/torque innovation (and the latent innovation); then innovation-triggered re-query with a max-interval fallback | queued |

## Later, independent of the chain

- **VLM latents (was EXP-0001 A1b/A2):** FAST fine-tune of `pi05_base`, then the student setup with a real VLM latent instead of the teacher. Probe every candidate z for continuous and future state first (ALPS-style probe, Theme 8), including V-JEPA 2 as a slow-side encoder (H0).
- **State hypotheses (was EXP-0001 Phase B):** tasks where the needed state is unobservable from proprio and force/torque and changes during the episode (observability rule; object moved while not touching it, moving targets).
- **Infrastructure:** backbone-only latency on an H200, the Jetson Orin NX/Nano benchmark, full-vision eval timing.

## Sources for the mechanisms

- Plan + feedback, delay handling, innovation gating, event triggers: `literature/themes/07-learned-filtering-in-policies.md` (Sec. 1 and 3).
- Latent predictors and JEPA: `literature/themes/08-predictive-latents-jepa.md` (Sec. 1 and 3).
- The retired all-in-one draft ("A1-teacher v2", 9 variants) is in git history (commit 43cf56e); the experiments above split it into single questions.
