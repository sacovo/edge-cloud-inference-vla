#!/bin/bash
# EXP-0001 A1: per-task training arrays (prepare.py: sensor windows, object poses, gripper contacts), one task per element.
# Usage: ls $ECVLA_SCRATCH/phaseA/*/*.h5 > manifest; sbatch --array=0-29 scripts/slurm/a1_prepare.sbatch <manifest> <out_dir> [extra]
# Test first (Object = heaviest scenes): sbatch -p cpu-debug -t 00:20:00 --array=<object idx> ... <manifest> <test_out> --max-demos 2
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G  # same hard-reset memory growth as the quality job
#SBATCH --time=02:00:00
#SBATCH --job-name=ecvla-a1-prepare
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; OUT_DIR=$2; shift 2
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
SRC=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
SUITE=$(basename "$(dirname "$SRC")"); TASK=$(basename "$SRC" .h5)
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID src=$SRC"
$ECVLA_RUN python -m ecvla.prepare "$SRC" "$ECVLA_SCRATCH/phaseA_quality/$SUITE/$TASK.h5" "$OUT_DIR/$SUITE/$TASK.npz" "$@"
