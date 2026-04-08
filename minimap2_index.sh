#!/bin/bash
#SBATCH -J minimap2_array
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-13%4
#SBATCH -o %x.%A_%a.out
#SBATCH -e %x.%A_%a.err
#SBATCH -A r00270

set -eo pipefail

# Environment
module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
conda activate f5c_env

# Paths
BASE="/N/project/NGS-JangaLab/Matthew/rna_seq_data/raw_data/HEK293T_RNA004"
CHUNK_DIR="$BASE/fastq/fastq_chunks"
OUTDIR="$BASE/bam_chunks"
TMP_BASE="$BASE/tmp_sort"
REF_FA="/N/project/NGS-JangaLab/Matthew/rna_seq_data/reference/GRCh38.primary_assembly.genome.fa"

mkdir -p "$OUTDIR" "$TMP_BASE"

# Input chunk
FASTQ_FILE=$(ls "$CHUNK_DIR"/chunks_*.fastq | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
OUT_BAM="$OUTDIR/chunk_${SLURM_ARRAY_TASK_ID}.bam"

if [[ ! -f "$FASTQ_FILE" ]]; then
    echo "[ERROR] Missing FASTQ: $FASTQ_FILE"
    exit 1
fi

echo "[INFO] Processing: $FASTQ_FILE"

# Thread control
THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$THREADS

# Temp directory
TMP_DIR="$TMP_BASE/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP_DIR"

# Alignment + Sorting
minimap2 -t $THREADS -ax splice -uf -k14 "$REF_FA" "$FASTQ_FILE" \
| samtools sort \
    -@ $THREADS \
    -m 1G \
    -T "$TMP_DIR/sort" \
    -o "$OUT_BAM"

# Index BAM
samtools index "$OUT_BAM"

# Cleanup
rm -rf "$TMP_DIR"

echo "[INFO] Finished chunk ${SLURM_ARRAY_TASK_ID}"