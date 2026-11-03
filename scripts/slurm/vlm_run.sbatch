#!/bin/bash
# Run one Python script in the EXP-0005 VLM env with the project on the path, on one GPU (EGL rendering, CUDA).
# For short one-off checks; use instead of `sbatch --wrap`, whose /bin/sh does not run `source` reliably.
# Usage: sbatch [-t 00:30:00] scripts/slurm/vlm_run.sbatch <script.py> [args]   (absolute paths in args)
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=ecvla-vlm-run
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
echo "git=$(git rev-parse HEAD) $*"
PYTHONPATH=src $VLM_RUN python "$@"
