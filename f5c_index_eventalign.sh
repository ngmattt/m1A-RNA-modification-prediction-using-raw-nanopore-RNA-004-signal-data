#!/bin/bash -l
#SBATCH -J f5c_eventalign
#SBATCH -o f5c_eventalign_%A_%a.out
#SBATCH -e f5c_eventalign_%A_%a.err
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngmat@iu.edu
#SBATCH -A r00270

set -euo pipefail

# ----------------------------
# 0) Environment
# ----------------------------
module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
conda activate f5c_env
THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "[info] Tools:"
command -v f5c   || true; f5c --version   || true
command -v minimap2 || true; minimap2 --version || true
command -v samtools || true; samtools --version || true
command -v slow5tools || true

# ----------------------------
# 1) Paths
# ----------------------------
BASE="/N/project/NGS-JangaLab/Matthew/rna_seq_data"
READS_FASTQ="$BASE/raw_data/HEK293T_RNA004/fastq/SGNex_Hek293T_directRNA_replicate5_run1.fastq.gz"
BLOW5="$BASE/raw_data/HEK293T_RNA004/blow5/SGNex_Hek293T_directRNA_replicate5_run1.blow5"
BAM="$BASE/raw_data/HEK293T_RNA004/bam/SGNex_Hek293T_directRNA_replicate5_run1_sorted.bam"
REFDIR="$BASE/reference"
REFERENCE_FA="$REFDIR/GRCh38.primary_assembly.genome.fa"
MODEL="$REFDIR/rna004.nucleotide.5mer.model"

OUT_DIR="$BASE/f5c_output"

mkdir -p "$OUT_DIR" "${SLURM_TMPDIR:-/tmp}"

# ----------------------------
# 2) f5c index
# ----------------------------
echo "[info] f5c index for: $BLOW5"
f5c index --slow5 "$BLOW5" -t "$THREADS" "$READS_FASTQ"

# ----------------------------
# 3) f5c eventalign (per-batch)
# ----------------------------
OUT_TSV="$OUT_DIR/$(basename "$BLOW5")_genome.eventalign.tsv"
echo "[info] Running f5c eventalign -> $OUT_TSV"

f5c eventalign \
  --rna \
  --pore rna004 \
  --reads "$READS_FASTQ" \
  --bam "$BAM" \
  --slow5 "$BLOW5" \
  --genome "$REFERENCE_FA" \
  --kmer-model "$MODEL" \
  --scale-events \
  --signal-index \
  --samples \
  --min-mapq 0 \
  --min-recalib-events 100 \
  --skip-unreadable=yes \
  --secondary=no \
  -t "$THREADS" \
  -K 4096 \
  -o "$OUT_TSV"

echo "[info] Done. Output: $OUT_TSV"



