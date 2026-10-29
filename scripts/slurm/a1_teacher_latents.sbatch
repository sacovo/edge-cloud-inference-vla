#!/bin/bash
# EXP-0001 A1-teacher: precompute the teacher's z (action chunk + hidden state) on all demo frames of one suite.
# Manifest lines: "<teacher_run_dir> <prepared_suite_dir> <out_dir>" ($ECVLA_SCRATCH is expanded). Skips finished files.
# Usage: sbatch --array=0-N scripts/slurm/a1_teacher_latents.sbatch <manifest>
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=ecvla-a1-teacher-z
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; shift
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
eval "set -- $LINE"
echo "host=$(hostname) teacher=$1 data=$2 out=$3"
$ECVLA_RUN python -m ecvla.teacher "$1" "$2" "$3"
