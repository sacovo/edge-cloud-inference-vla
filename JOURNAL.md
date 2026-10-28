# Research Journal

Newest entries first. Each entry covers what was tried, the result, and what failed and why.

## 2026-10-28: EXP-0001 decisions (slow model, demos, onboard profile)

- Slow model: `pi05_base` only, not `pi05_libero`, which was fine-tuned on LIBERO and would skew z. Added PaliGemma 2 28B as the generic ~27B remote model.
- The author noted that π0.5 (~3B) fits on a Jetson but is too slow to output actions for control. That leads to a second deployment profile: both loops on the Jetson, with the 3B model as a slow loop emitting only z and a tiny fast policy on top. This is the same research question with a different latency model (fixed compute delay instead of network jitter), so it went into EXP-0001 as profile P-onboard rather than a new experiment.
- Demos: teleop a few seed episodes, train ACT or SmolVLA, keep successful rollouts, retrain, and add manual episodes where needed. The author's RECAP-on-SmolVLA work is an option for improving the collector. Watch for success-filter bias toward easy states and too few recoveries.

## 2026-10-27: EXP-0001 designed (task-ID vs state crossover)

- Scope decision: manipulators are the primary platform, because training data and sim environments are plentiful. Drones were one example and are now optional (E4).
- Author's hypothesis: a large fast policy uses the MLLM latent only as a task ID. Below some size the latent has to carry scene state, the task-ID substitute fails, and staleness starts to matter.
- Author's key ablation: a fully blind fast policy (proprio + torque + wrist F/T). All scene state must then come through z.
- Added a delayed-oracle condition, so that "z lacks state" can be told apart from "state too old".
- Design fixed in advance in `experiments/EXP-0001-latent-taskid-vs-state/README.md`: hypotheses H0–H3, pass/fail criteria, and stages S0–S3. S1 is an oracle-only pilot with no VLM, which checks that the tasks need scene state before any VLM compute is spent.
- Next: S0. Resolve the open decisions (which VLM, how to generate demos for the event/dynamic tasks, LIBERO defaults) and build the delay/multi-rate env wrapper.

## 2026-10-27: Rapid scoping review completed

- Ran the review per `literature/PROTOCOL.md`: 5 themes, each searched by its own agent. The result is 147 deduplicated references, with every arXiv ID checked against the arXiv API.
- Closest prior work: AsyncVLA (arXiv 2602.13476), CloudEdgeVLA (arXiv 2608.00569), Figure Helix. Key gaps:
  - no fast-policy size sweep under a VLM latent
  - no explicit latent-age input under jitter
  - no fusion of IMU, torque, camera and a remote latent
  - no drone platform
  - no slow model around 27B
- Main threat to the experiments: the latent may encode only which task is being done (OpenHelix). The size sweep needs no-latent and task-ID-only baselines.
- Tooling: added a hook that verifies the bibliography. It caught that a wrong title with a reused arXiv ID was silently dropped as a duplicate during the merge, so `verify_bib.py` now checks the per-theme source files directly.
- Next: E0 from `literature/SYNTHESIS.md` §5. Pick the simulator and the 27B VLM, then precompute embeddings for LIBERO.
