#!/bin/bash
#SBATCH --job-name=m1A_rf
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rf_baseline_%j.out
#SBATCH --error=rf_baseline_%j.err
#SBATCH -A r00270

set -eo pipefail

module load conda
conda activate ml_env

# -----------------------------
# Run training
# -----------------------------
SCRIPT="/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/train_baseline_xgb.py"

echo "Starting Random Forest training"

python "${SCRIPT}"

echo "Training finished"
