# Research Journal

Newest entries first. Each entry covers what was tried, the result, and what failed and why.

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
