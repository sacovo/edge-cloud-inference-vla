# Edge–cloud VLA: slow remote VLM → latent → small fast on-robot policy

Research question: how small can the on-robot policy be when it is conditioned on an embedding from a ~27B model on a stationary machine (~5 Hz, network latency and jitter)? The on-robot policy fuses IMU ~100 Hz, torque ~60 Hz and camera ~20 Hz, outputs actions at ≥20 Hz, and runs on a battery/weight-limited robot or drone.

## Current phase
- Phase 1 (done 2026-10-27): rapid scoping review. Start with `literature/SYNTHESIS.md`; protocol in `literature/PROTOCOL.md`.
- Phase 2 (next): sim experiments E0–E5 from `literature/SYNTHESIS.md` §5. No code or environment exists yet.

## Session start
Read `JOURNAL.md` (latest entries first) before doing anything else.

## Commands
- Rebuild bibliography: `python3 scripts/merge_bib.py`
- Verify bibliography: `python3 scripts/verify_bib.py` (must exit 0; queries the arXiv API)
- A PostToolUse hook (`scripts/hooks/bib_check.sh`) runs both whenever `literature/bib/*.bib` is edited. If it reports a MISMATCH, fix the entry; do not weaken the check.

## Citations (non-negotiable)
- Never write a BibTeX entry, arXiv ID, author list or number from memory. Fetch it from arXiv/DOI/the project page first.
- Edit `literature/bib/<theme>.bib`. `literature/references.bib` is generated: never edit it directly.
- Every claim in `literature/` carries a citekey or arXiv ID plus a locator (section/table/figure). If you only read the abstract, say so ("abstract only").
- Unverifiable sources go in the theme file's "Unverified / excluded" list, never in the bibliography.
- Paper read status is the author's to declare (`/ars-mark-read`), never inferred.

## Experiments (once Phase 2 starts)
- Before running: write `experiments/EXP-NNNN-<slug>/README.md` with the hypothesis and a success criterion fixed in advance.
- Use ≥3 seeds for any claim. Log git SHA, config and seed. Record failed and crashed runs too.
- Never edit evaluation code or held-out task lists while optimising a metric. Ask first.
- Run long jobs in tmux or the background, never in the foreground.
- After each run or session, append to `JOURNAL.md`: what was tried, the result, and why it failed if it did.

## Gotchas
- Many 2025–26 numbers in `literature/themes/` came from automated full-text extraction. Re-check against the PDF before quoting them in a paper.
- Theme 4 flags Chronos Policy Fig. 3 per-condition numbers as possibly inconsistent; use only its qualitative conclusions.
