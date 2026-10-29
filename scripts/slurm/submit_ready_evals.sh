#!/bin/bash
# Submit eval lines (first field = run dir) whose run has model.pt and that were not submitted yet; remembers
# what it submitted in <manifest>.submitted. Only file checks + sbatch, so it is safe on the login node or in a job.
# Usage: scripts/slurm/submit_ready_evals.sh <eval_manifest>
set -euo pipefail
cd ~/work/edge-cloud-inference-vla
export ECVLA_SCRATCH=/scratch/sandro.covo/ecvla
ALL=$1; DONE=${ALL%.txt}.submitted; touch "$DONE"
NEW=$(mktemp -p $ECVLA_SCRATCH/a1 batch_XXXX.txt)
while read -r run rest; do
  grep -qxF "$run $rest" "$DONE" && continue
  [ -f "$(eval echo $run)/model.pt" ] && echo "$run $rest" >> "$NEW"
done < "$ALL"
n=$(wc -l < "$NEW")
if [ "$n" -gt 0 ]; then
  sbatch --array=0-$((n - 1))%40 scripts/slurm/a1_eval.sbatch "$NEW" && cat "$NEW" >> "$DONE"
fi
echo "submitted=$n total_submitted=$(wc -l < "$DONE") of $(wc -l < "$ALL")"
