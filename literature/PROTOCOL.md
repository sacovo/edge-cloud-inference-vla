# Rapid Scoping Review — Protocol

- **Date started:** 2026-10-27
- **Form:** rapid scoping review, chosen by the author. It maps the field and does not assess effects. Shortcuts: one screener (AI-assisted), no dual screening, and no grey-literature search beyond major lab blogs and tech reports. The PRISMA-ScR checklist is the author's responsibility.
- **Tooling:** Claude Code + the ARS `deep-research` plugin in `lit-review` mode, with parallel AI search agents. Every reference must resolve to a real arXiv ID, DOI, or official project page, or it is dropped.

## System under study

A **slow model** runs on a stationary compute box (~27B VLM/VLA, ~5 Hz, network latency and jitter). It emits a latent or embedding that conditions a **fast model** on a mobile robot with tight battery and weight limits, possibly a drone. The fast model fuses multi-rate sensors (IMU ~100 Hz, camera ~20 Hz, joint torque ~60 Hz) and outputs action sequences at ≥20 Hz.

## Research questions

- **RQ1:** What architectures split robot policies into slow, large reasoning models and fast, small control models? How is information passed between them (latent token, language, waypoints, action prior)?
- **RQ2:** How does prior work handle latency, asynchrony and staleness between the two levels and between perception and action (action chunking, async inference, real-time chunking, delay-aware training)?
- **RQ3:** How small can on-device policies get (parameter count, FLOPs, Hz, watts) on embedded hardware (Jetson, MCU, drones), and what do they lose relative to large models?
- **RQ4:** How do learned policies fuse sensors that arrive at different rates (IMU, vision, force/torque, proprioception)?
- **RQ5:** What evidence is there on how much a conditioning embedding can shrink a downstream policy (distillation, privileged teacher–student, latent-conditioned control, compressing what is sent over the link)? Which benchmarks or sim setups suit experiments?

## Eligibility

- **Include:** papers and tech reports from 2015 onward (older seminal work allowed) on robot learning, embodied AI, edge/cloud robotics, or learned control, if they address at least one RQ. arXiv preprints count, since the field is preprint-heavy, and are flagged as Tier 2.
- **Exclude:** pure NLP or LLM serving work with no embodiment; non-learned classical control, unless used as a baseline reference; sources that cannot be verified.

## Sources searched

arXiv, Semantic Scholar, Google Scholar (via web search), project pages, and lab blogs (Physical Intelligence, Figure, NVIDIA, Google DeepMind, Hugging Face).

## Outputs

| File | Content |
|---|---|
| `themes/*.md` | Annotated bibliography per theme (RQ1–RQ5) |
| `references.bib` | BibTeX for all included sources |
| `SYNTHESIS.md` | Cross-theme synthesis, evidence map, gaps, implications for experiments |
| `SEARCH_LOG.md` | Search strings and counts per theme |
