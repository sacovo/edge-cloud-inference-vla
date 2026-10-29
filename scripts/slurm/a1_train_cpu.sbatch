#!/bin/bash
# EXP-0001 A1: train small fast policies (10k / 100k head params) on CPU nodes, which are mostly idle while GPUs
# are scarce. Same manifest format as a1_train.sbatch; finished runs (model.pt present) are skipped.
# Usage: sbatch --array=<indices>%K scripts/slurm/a1_train_cpu.sbatch <train_manifest> [extra args]
# Test first: sbatch -p cpu-debug -t 00:20:00 --array=<idx> ... <manifest> --steps 500 --out <test_dir>
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=ecvla-a1-train-cpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; shift
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
eval "set -- $LINE \"\$@\""
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID args=$*"
$ECVLA_RUN python -m ecvla.train --device cpu "$@"
