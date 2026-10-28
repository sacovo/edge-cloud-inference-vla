#!/bin/bash
# EXP-0001 A0: build the Phase A dataset, one LIBERO task file per array element.
# Usage: sbatch --array=0-N%K scripts/slurm/a0_build_dataset.sbatch <manifest.txt> <out_dir> [extra args]
# Test first: sbatch -p cpu-debug -t 00:20:00 --array=0 ... <manifest> <test_out_dir> --max-demos 2
# The manifest lists one source hdf5 path per line. Finished outputs are skipped, so reruns are safe.
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=ecvla-a0-build
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; OUT_DIR=$2; shift 2
SRC=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
SUITE=$(basename "$(dirname "$SRC")")
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID src=$SRC"
$ECVLA_RUN python -m ecvla.build_dataset "$SRC" "$OUT_DIR/$SUITE/$(basename "$SRC" .hdf5).h5" "$@"
