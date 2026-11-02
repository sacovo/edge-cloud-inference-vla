#!/bin/bash
# Build the EXP-0005 VLM env (vlm/) on a compute node and check imports.
# Usage: sbatch scripts/slurm/vlm_setup.sbatch
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=ecvla-vlm-setup
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
hostname
uv python install 3.12
uv sync --project vlm
$VLM_RUN python -c "import torch, lerobot, transformers; from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy; print('IMPORT-OK', torch.__version__, torch.version.cuda, lerobot.__version__, transformers.__version__)"
