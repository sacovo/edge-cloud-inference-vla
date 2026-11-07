# Environment for EXP-0005 slow-model (VLM) jobs: the separate uv project in vlm/ (LeRobot, torch 2.11.0+cu126).
# Source it from sbatch scripts after `cd "$SLURM_SUBMIT_DIR"`.
source scripts/slurm/env.sh
VLM_RUN="uv run --project vlm"
# HF_HOME points at /scratch (env.sh), but the login token lives in the default location; read it from there
# (gated repos: google/paligemma-3b-pt-224).
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
# GPU assignment, logged for diagnosis (exit-134 aborts on calc-g-041 since 2026-11-07 ~06:00, see JOURNAL).
echo "gpus: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} host=$(hostname)"
