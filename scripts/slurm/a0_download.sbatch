#!/bin/bash
# Download the original LIBERO HDF5 suites to scratch and write the manifest.
# Usage: sbatch scripts/slurm/a0_download.sbatch [suite ...]
#SBATCH --clusters=calc-cpu
#SBATCH --partition=cpu-debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --job-name=ecvla-a0-download
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/slurm/env.sh
scripts/a0_download_libero.sh "$ECVLA_SCRATCH/raw" "$ECVLA_SCRATCH/raw/manifest_phaseA.txt" "$@"
du -sh "$ECVLA_SCRATCH/raw"
