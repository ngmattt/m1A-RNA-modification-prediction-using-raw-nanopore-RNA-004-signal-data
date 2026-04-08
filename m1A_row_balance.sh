#!/bin/bash
#SBATCH -J row_balance_m1A
#SBATCH -o row_balance_%j.out
#SBATCH -e row_balance_%j.err
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH --time=04:00:00
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -A r00270

set -euo pipefail
export PS1=""

# ----------------------------
# Environment
# ----------------------------
module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh

# (optional) activate env if needed
# conda activate ml_env

echo "[INFO] Starting row-level balancing..."

# ----------------------------
# Paths
# ----------------------------
WORKDIR="/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data"
INPUT="$WORKDIR/m1A_site_balanced1.tsv.gz"
OUTFILE="$WORKDIR/m1A_fully_balanced.tsv"

cd "$WORKDIR"

# ----------------------------
# Step 1: Split positives and negatives
# ----------------------------
echo "[INFO] Splitting positives and negatives..."

zcat "$INPUT" | awk 'NR==1 || $NF==1' > pos.tmp
zcat "$INPUT" | awk '$NF==0' > neg.tmp

# ----------------------------
# Step 2: Count negatives (target size)
# ----------------------------
NEG_COUNT=$(grep -c -v "^label" neg.tmp)
echo "[INFO] Negative count: $NEG_COUNT"

# ----------------------------
# Step 3: Sample positives
# ----------------------------
echo "[INFO] Sampling positives to match negatives..."

shuf -n "$NEG_COUNT" pos.tmp > pos_sampled.tmp

# ----------------------------
# Step 4: Clean headers
# ----------------------------
echo "[INFO] Cleaning headers..."

head -n 1 pos.tmp > header.tmp
grep -v "^label" pos_sampled.tmp > pos_clean.tmp
grep -v "^label" neg.tmp > neg_clean.tmp

# ----------------------------
# Step 5: Combine final dataset
# ----------------------------
echo "[INFO] Combining final dataset..."

cat header.tmp pos_clean.tmp neg_clean.tmp > "$OUTFILE"

# ----------------------------
# Step 6: Compress
# ----------------------------
echo "[INFO] Compressing output..."

gzip -f "$OUTFILE"

# ----------------------------
# Step 7: Verify balance
# ----------------------------
echo "[INFO] Verifying row-level balance..."

zcat "${OUTFILE}.gz" | awk '{print $NF}' | sort | uniq -c

echo "[INFO] Done!"