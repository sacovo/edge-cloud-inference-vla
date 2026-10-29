#!/bin/bash
# EXP-0001 A1: frozen ResNet-18 features for `full` vision, all tasks of a manifest in one GPU job (skips finished files).
# Usage: sbatch scripts/slurm/a1_vision.sbatch <manifest> <out_dir> [extra]
# Test first: sbatch -p debug -t 00:20:00 scripts/slurm/a1_vision.sbatch <manifest> <test_out> --max-demos 2
#SBATCH --clusters=cluster
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=ecvla-a1-vision
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
MANIFEST=$1; OUT_DIR=$2; shift 2
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
$ECVLA_RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
while read -r SRC; do
  SUITE=$(basename "$(dirname "$SRC")"); TASK=$(basename "$SRC" .h5)
  $ECVLA_RUN python -m ecvla.vision "$SRC" "$OUT_DIR/$SUITE/$TASK.npz" "$@"
done < "$MANIFEST"
