# Environment for EXP-0005 slow-model (VLM) jobs: the separate uv project in vlm/ (LeRobot, torch 2.11.0+cu126).
# Source it from sbatch scripts after `cd "$SLURM_SUBMIT_DIR"`.
source scripts/slurm/env.sh
VLM_RUN="uv run --project vlm"
