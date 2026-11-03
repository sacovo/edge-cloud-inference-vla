#!/bin/bash
# EXP-0005 slow-only check: a fine-tuned pi0_fast acts alone in the LIBERO sim (scripts/exp0005_slow_only.py). One GPU.
# Usage: sbatch scripts/slurm/exp0005_slow_only.sbatch --ckpt <.../pretrained_model> --mixture <dir> --suite <s> \
#          (--task N | --task-key <bddl name>) --episodes 2 --out <json>   (absolute paths)
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=ecvla-vlm-slow
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
echo "git=$(git rev-parse HEAD)"
PYTHONPATH=src $VLM_RUN python scripts/exp0005_slow_only.py "$@"
