#!/bin/bash
# Cluster-side eval auto-submitter: every 15 min, submit evals for runs whose training finished
# (submit_ready_evals.sh), until every manifest is fully submitted or the time limit is reached.
# Runs on the cluster so a dropped VPN/SSH connection does not stall the pipeline.
# Usage: sbatch scripts/slurm/a1_submitter.sbatch <eval_manifest> [<eval_manifest> ...]
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-daily
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=ecvla-submitter
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
while true; do
  left=0
  for m in "$@"; do
    out=$(scripts/slurm/submit_ready_evals.sh "$m" 2>&1 | tail -1)
    echo "$(date +%F_%H:%M) $(basename "$m"): $out"
    total=$(wc -l < "$m"); done_=$(wc -l < "${m%.txt}.submitted")
    [ "$done_" -lt "$total" ] && left=1
  done
  [ "$left" = 0 ] && { echo "all submitted"; exit 0; }
  sleep 900
done
