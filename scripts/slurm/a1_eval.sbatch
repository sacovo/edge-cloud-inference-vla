#!/bin/bash
# EXP-0001 A1: closed-loop LIBERO evaluation, one (run, task, k) per array element; manifest lines are
# "<run_dir> <suite> <task_id> <k|->" ($ECVLA_SCRATCH is expanded). Results go to <run_dir>/eval/; finished ones skip.
# Usage: sbatch --array=0-N%K [--dependency=afterok:<train job>] scripts/slurm/a1_eval.sbatch <eval_manifest> [extra]
# Test first: sbatch -p cpu-debug -t 00:20:00 --array=0 ... <manifest> --episodes 2 --out-suffix _test
# Blind runs need no rendering (~5 s per 220-step episode locally); `full` vision renders 2 cameras at 224 px.
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=ecvla-a1-eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; shift
SUFFIX=""
if [[ "${1:-}" == "--out-suffix" ]]; then SUFFIX=$2; shift 2; fi
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
eval "set -- $LINE \"\$@\""
RUN=$1; SUITE=$2; TASK=$3; K=$4; shift 4
KARG=(); [[ "$K" != "-" ]] && KARG=(--k "$K")
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID run=$RUN suite=$SUITE task_id=$TASK k=$K"
$ECVLA_RUN python -m ecvla.evaluate "$RUN" --suite "$SUITE" --task "$TASK" "${KARG[@]}" \
  --out "$RUN/eval$SUFFIX/${SUITE}_t${TASK}_k${K}.json" "$@"
