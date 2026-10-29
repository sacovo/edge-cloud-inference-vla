#!/bin/bash
# EXP-0001 A1: closed-loop LIBERO evaluation of one trained run per array element: all 10 tasks of its suite (and
# every k for oracle-delayed) in one process. Manifest lines: "<run_dir> <suite> <k list, e.g. 1,10 | ->"
# ($ECVLA_SCRATCH is expanded). Results go to <run_dir>/eval/<suite>_t<task>_k<k>.json; finished ones are skipped,
# so a rerun only does what is missing. One job per run keeps the job count low (144 for the blind grid half).
# Usage: sbatch --array=0-N%K scripts/slurm/a1_eval.sbatch <eval_manifest> [extra evaluate args]
# Test first: sbatch -p cpu-debug -t 00:20:00 --array=0 ... <manifest> --task 0 --episodes 2
# Time: blind runs ~5-10 s per episode -> ~1-1.5 h per run, ~2-3 h with two k values.
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=ecvla-a1-eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
MANIFEST=$1; shift
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
eval "set -- $LINE \"\$@\""
RUN=$1; SUITE=$2; K=$3; shift 3
KARG=(); [[ "$K" != "-" ]] && KARG=(--k ${K//,/ })
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID run=$RUN suite=$SUITE k=$K"
$ECVLA_RUN python -m ecvla.evaluate "$RUN" --suite "$SUITE" --task 0 1 2 3 4 5 6 7 8 9 "${KARG[@]}" \
  --out-dir "$RUN/eval" "$@"
