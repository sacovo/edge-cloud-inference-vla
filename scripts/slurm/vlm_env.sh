# Environment for EXP-0005 slow-model (VLM) jobs: the separate uv project in vlm/ (LeRobot, torch 2.11.0+cu126).
# Source it from sbatch scripts after `cd "$SLURM_SUBMIT_DIR"`.
source scripts/slurm/env.sh
VLM_RUN="uv run --project vlm"
# HF_HOME points at /scratch (env.sh), but the login token lives in the default location; read it from there
# (gated repos: google/paligemma-3b-pt-224).
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
# MuJoCo/robosuite EGL picks its device from CUDA_VISIBLE_DEVICES, which Slurm renumbers from 0 inside the job, while
# EGL lists every GPU on the node: a job on physical GPU k > 0 then rendered on another job's GPU and aborted (exit
# 134, no traceback; jobs 279628, 279766, 279767 on calc-g-041). Render on the GPU Slurm assigned instead.
if [ -n "${SLURM_JOB_GPUS:-}" ] && [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]; then
  export MUJOCO_EGL_DEVICE_ID=${SLURM_JOB_GPUS%%,*}
fi
echo "gpus: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
