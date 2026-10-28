#!/bin/bash
# Build the project venv on a compute node and verify it (imports, rendering via tests).
# Usage: sbatch scripts/slurm/setup_env.sbatch
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ecvla-setup
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
hostname
uv python install 3.12
uv sync --extra libero --extra dev
$ECVLA_RUN python -c "import robosuite, mujoco, libero; print('IMPORT-OK', robosuite.__version__, mujoco.__version__)"
$ECVLA_RUN pytest tests -q -rs
