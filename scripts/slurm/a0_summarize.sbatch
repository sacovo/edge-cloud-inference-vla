#!/bin/bash
# Summarize the built Phase A dataset. Usage: sbatch scripts/slurm/a0_summarize.sbatch <dataset_dir> <out.json>
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=ecvla-a0-summary
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
uv run python scripts/a0_summarize_dataset.py "$1" --out "$2"
