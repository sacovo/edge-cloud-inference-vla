# Edge–cloud VLA: slow remote VLM → latent → small fast on-robot policy

Research question: how small can the on-robot policy be when it is conditioned on an embedding from a ~27B model on a stationary machine (~5 Hz, network latency and jitter)? The on-robot policy fuses IMU ~100 Hz, torque ~60 Hz and camera ~20 Hz, outputs actions at ≥20 Hz, and runs on a battery/weight-limited robot or drone.

## Current phase
- Phase 1 (done 2026-10-27): rapid scoping review. Start with `literature/SYNTHESIS.md`; protocol in `literature/PROTOCOL.md`.
- Phase 2 (next): sim experiments, manipulators first. EXP-0001 (`experiments/EXP-0001-latent-taskid-vs-state/README.md`) is designed; its criteria are pre-registered, so log any change in its Change log. A0 has started: the replay check passed (see JOURNAL).

## Session start
Read `JOURNAL.md` (latest entries first) before doing anything else.

## Commands
- Env: `uv sync --extra libero --extra dev` (Python ≤3.12; hf-libero 0.1.4, robosuite 1.4.0). Unset `VIRTUAL_ENV` if uv warns about it. Local rendering uses `MUJOCO_GL=egl` (Mesa/ROCm); `src/ecvla/libero_env.py` writes a project-local LIBERO config. Never edit `~/.libero`.
- LIBERO replay check: `uv run python scripts/a0_replay_check.py <demo.hdf5> --episodes 5`. Original HDF5s with sim states: `yifengzhu-hf/LIBERO-datasets` (not `HuggingFaceVLA/libero`, which has no states). Local data in `data/` (gitignored).
- HPC: `ssh sandro.covo@calculon.informatik.fhnw.ch`, Slurm; data on `/scratch/sandro.covo/ecvla`, HF cache on `/scratch/sandro.covo/hf`. No conda.
  - **Login node: git/sbatch/squeue only.** Installs (`sbatch scripts/slurm/setup_env.sbatch`), downloads and sims run as jobs.
  - Before a large job, test the same sbatch script on `cpu-debug` with a tiny input.
  - Job env: `scripts/slurm/env.sh` (uv-managed Python, `--extra libero`).
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
