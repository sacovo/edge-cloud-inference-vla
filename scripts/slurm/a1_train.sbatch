#!/bin/bash
# EXP-0001 A1: train one fast policy per array element; each manifest line holds the ecvla.train arguments
# ($ECVLA_SCRATCH is expanded). Finished runs (model.pt present) are skipped.
# Usage: sbatch --array=0-N%K scripts/slurm/a1_train.sbatch <train_manifest> [extra args, e.g. --steps 200]
# Test first: sbatch -p debug -t 00:20:00 --array=<largest model idx> ... <manifest> --steps 200 --out <test_dir>
#SBATCH --clusters=cluster
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ecvla-a1-train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; shift
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
eval "set -- $LINE \"\$@\""
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID args=$*"; nvidia-smi -L
$ECVLA_RUN python -m ecvla.train --require-gpu "$@"
