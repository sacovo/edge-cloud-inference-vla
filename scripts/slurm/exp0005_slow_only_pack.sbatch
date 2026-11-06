#!/bin/bash
# EXP-0005 slow-only, several tasks as parallel processes on one H200 (one fp32 policy ~17 GB; the h200 queue is
# the bottleneck, not the GPU). Same script and arguments as exp0005_slow_only.sbatch, one output JSON per task.
# Usage: sbatch scripts/slurm/exp0005_slow_only_pack.sbatch <out dir> "<suite>:<task> ..." <args for exp0005_slow_only.py>
#   (args without --suite/--task/--out; absolute paths)
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=ecvla-vlm-slowpack
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
OUT=$1; ITEMS=$2; shift 2
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
echo "git=$(git rev-parse HEAD) items=$ITEMS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=()
for it in $ITEMS; do
  su=${it%%:*}; t=${it##*:}
  PYTHONPATH=src $VLM_RUN python scripts/exp0005_slow_only.py --suite "$su" --task "$t" --out "$OUT/${su}_t$t.json" "$@" \
    > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${su}_t$t.out" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
exit $rc
