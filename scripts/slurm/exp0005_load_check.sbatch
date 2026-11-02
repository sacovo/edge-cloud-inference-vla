#!/bin/bash
# EXP-0005 stage 0: load pi05_base's VLM into LeRobot's pi0_fast and round-trip LIBERO chunks through FAST+ (CPU).
# Downloads lerobot/pi05_base (~14 GB, F32) into the HF cache on first run. Needs an HF token with the PaliGemma
# license accepted (the pi0_fast init loads google/paligemma-3b-pt-224's tokenizer).
# Usage: sbatch scripts/slurm/exp0005_load_check.sbatch
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=ecvla-vlm-load
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
mkdir -p $ECVLA_SCRATCH/exp0005
$VLM_RUN python scripts/exp0005_load_check.py --out $ECVLA_SCRATCH/exp0005/load_check.json
