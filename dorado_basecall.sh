#!/bin/bash
#SBATCH --job-name=dorado_basecall
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=dorado_%j.out
#SBATCH --error=dorado_%j.err
#SBATCH -A r00270

set -eo pipefail

# ----------------------------
# Environment
# ----------------------------
module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
conda activate f5c_env

# Dorado binary
export DORADO_HOME=/N/project/NGS-JangaLab/Matthew/rna_seq_data/dorado-0.7.2-linux-x64
export PATH="$DORADO_HOME/bin:$PATH"

echo "[INFO] Dorado version:"
dorado --version

echo "[INFO] GPUs visible:"
nvidia-smi

# ----------------------------
# Paths
# ----------------------------
BASE="/N/project/NGS-JangaLab/Matthew/rna_seq_data"
POD5="$BASE/raw_data/HEK293T_RNA004/pod5/SGNex_Hek293T_directRNA_replicate5_run1.pod5"
MODEL="$BASE/dorado_models/rna004_130bps_sup@v5.0.0"
FASTQDIR="$BASE/raw_data/HEK293T_RNA004/fastq"
OUTDIR="$BASE/raw_data/HEK293T_RNA004/bam"

mkdir -p "$OUTDIR"

OUT_BAM="$OUTDIR/SGNex_Hek293T_directRNA_replicate5_run1.bam"
OUT_FASTQ="$FASTQDIR/SGNex_Hek293T_directRNA_replicate5_run1.fastq.gz"

# ----------------------------
# Basecalling (AUTOTUNED)
# ----------------------------
echo "[INFO] Starting Dorado basecalling"
echo "[INFO] POD5: $POD5"
echo "[INFO] Model: $MODEL"
echo "[INFO] Output: $OUT_BAM"

dorado basecaller \
  "$MODEL" \
  "$POD5" \
  --emit-moves \
  --device cuda:all \
  > "$OUT_BAM"

echo "[INFO] Basecalling finished"
ls -lh "$OUT_BAM"

samtools fastq "$OUT_BAM" | gzip > "$OUT_FASTQ"
ls -lh "$OUT_FASTQ"

samtools sort \
  -@ 16 \
  -o SGNex_Hek293T_directRNA_replicate5_run1_sorted.bam \
  "$OUT_BAM" 

samtools index \
  SGNex_Hek293T_directRNA_replicate5_run1_sorted.bam