#!/bin/bash
# EXP-0005: merge per-source mixture parts into one LeRobot dataset (scripts/exp0005_merge_mixture.py). CPU.
# Usage: sbatch [--dependency=afterok:<jobs>] scripts/slurm/exp0005_merge_mixture.sbatch --parts <dir> --sources ... \
#          --out <dir>   (absolute paths)
#SBATCH --clusters=cluster
#SBATCH --partition=performance
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=ecvla-vlm-merge
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
echo "git=$(git rev-parse HEAD)"
$VLM_RUN python scripts/exp0005_merge_mixture.py "$@"
