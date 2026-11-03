#!/bin/bash
# EXP-0005: FAST-token LoRA fine-tune of pi05_base's VLM in LeRobot's pi0_fast on a built mixture. One H200.
# Creates the start checkpoint (scripts/exp0005_make_init_ckpt.py) if missing, then runs lerobot_train with LoRA
# (r 16) on the language model's attention + MLP and the LM head trained in full (its FAST rows must be learned;
# EXP-0005 stage-0 loss test). Extra args go to lerobot_train (absolute paths: they are not expanded in the job).
# Usage: sbatch scripts/slurm/exp0005_train.sbatch <mixture dir> <run dir> [--steps 500 ...]
#SBATCH --clusters=cluster
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=ecvla-vlm-train
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
MIX=$1; RUN=$2; shift 2
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/vlm_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
INIT=$ECVLA_SCRATCH/exp0005/init/pi0fast_from_pi05base_$(basename "$MIX")
[ -f "$INIT/policy_preprocessor.json" ] || $VLM_RUN python scripts/exp0005_make_init_ckpt.py --mixture "$MIX" --out "$INIT"
echo "git=$(git rev-parse HEAD) mix=$MIX run=$RUN init=$INIT"
$VLM_RUN python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/$(basename "$MIX") --dataset.root="$MIX" \
  --policy.path="$INIT" --policy.push_to_hub=false --policy.device=cuda \
  --peft.method_type=LORA --peft.r=16 \
  --peft.target_modules='.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --peft.full_training_modules='["lm_head"]' \
  --output_dir="$RUN" --batch_size=16 --num_workers=6 --log_freq=10 --save_freq=500 --seed=0 "$@"
