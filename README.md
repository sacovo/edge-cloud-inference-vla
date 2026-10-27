# Edge–Cloud Split Inference for Robot Policies

**Question:** how small can an on-robot policy be when a larger model on a stationary machine sends it an embedding? The larger model is ~27B, runs at ~5 Hz and has network latency. The on-robot policy has to fuse IMU at ~100 Hz, torque at ~60 Hz and camera at ~20 Hz, and output actions at ≥20 Hz on a battery- and weight-limited robot or drone.

## Layout

| Path | Content |
|---|---|
| `literature/SYNTHESIS.md` | **Start here.** Cross-theme findings, gap list, threats to validity, proposed experiments E0–E5 |
| `literature/PROTOCOL.md` | Rapid scoping review protocol (RQ1–RQ5, eligibility) |
| `literature/themes/0N-*.md` | Annotated bibliographies per theme, with comparison tables |
| `literature/references.bib` | All references, deduplicated (generated) |
| `literature/bib/*.bib` | Per-theme BibTeX (source of truth; edit these) |
| `literature/SEARCH_LOG.md` | Queries used, for re-running the search |
| `docs/claude-workflow-best-practices.md` | How to run this project effectively with Claude Code |
| `scripts/merge_bib.py` | Rebuild `references.bib` from `literature/bib/` |
| `scripts/verify_bib.py` | Check every arXiv ID/title in `references.bib` against the arXiv API |
