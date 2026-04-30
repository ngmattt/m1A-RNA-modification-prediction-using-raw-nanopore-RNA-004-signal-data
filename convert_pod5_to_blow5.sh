#!/bin/bash
#SBATCH -J get_HEK293T_blow5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -A r00270

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f5c_env

POD5_DIR="/N/project/NGS-JangaLab/Matthew/rna_seq_data/raw_data/HEK293T_RNA004/SGNex_Hek293T_directRNA_replicate5_run1.pod5"
OUTDIR="/N/project/NGS-JangaLab/Matthew/rna_seq_data/raw_data/HEK293T_RNA004/blow5"

mkdir -p "$OUTDIR"

echo "[INFO] Converting split POD5 files to BLOW5 using blue-crab"

blue-crab p2s \
  "$POD5_DIR" \
  -d "$OUTDIR" \
  -c zlib \
  -p 4 \
  -t 8

echo "[INFO] Conversion complete"

