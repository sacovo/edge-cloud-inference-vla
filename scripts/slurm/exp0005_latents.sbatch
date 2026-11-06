#!/bin/bash
# EXP-0005 stage 1: R1 latents of the frozen fine-tuned pi0_fast on all LIBERO demo frames of one suite
# (scripts/exp0005_latents.py; renders each frame in our sim, so CPU-bound as well). One GPU.
# Usage: sbatch scripts/slurm/exp0005_latents.sbatch --ckpt <.../pretrained_model> --mixture <dir> \
#          --suite <s> --out-dir <dir> [--task N ...] [--max-demos N] [--dtype bfloat16]   (absolute paths)
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ecvla-vlm-latents
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "git=$(git rev-parse HEAD)"
PYTHONPATH=src $VLM_RUN python scripts/exp0005_latents.py "$@"
