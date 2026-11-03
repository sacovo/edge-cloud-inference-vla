#!/bin/bash
# EXP-0005: build the FAST fine-tuning mixture (scripts/exp0005_build_mixture.py) as one LeRobot dataset. CPU.
# Usage: sbatch [-p cpu-daily -t 08:00:00] scripts/slurm/exp0005_build_mixture.sbatch --sources libero utaustin_mutex \
#          --out /scratch/sandro.covo/ecvla/exp0005/datasets/<name> [--max-episodes N]
#   (absolute --out: extra args are not expanded inside the job)
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=ecvla-vlm-mix
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
$VLM_RUN python scripts/exp0005_build_mixture.py "$@"
