#!/bin/bash
# EXP-0005 stage 0: does pi05_base's LM head already predict FAST+ tokens? Teacher-forced pi0_fast loss on LIBERO
# frames for pi05_base (true / shuffled chunks) and plain PaliGemma (scripts/exp0005_fast_ce.py). One GPU.
# Usage: sbatch scripts/slurm/exp0005_fast_ce.sbatch [--n 64]
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=ecvla-vlm-ce
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p $ECVLA_SCRATCH/exp0005
$VLM_RUN python scripts/exp0005_fast_ce.py --out $ECVLA_SCRATCH/exp0005/fast_ce.json "$@"
