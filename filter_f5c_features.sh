#!/bin/bash
#SBATCH -J m1A_filter_features
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o Hek293T_m1A_filter_features.%j.out
#SBATCH -e Hek293T_m1A_filter_features.%j.err
#SBATCH -A r00270

set -euo pipefail

IN="/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/SGNex_Hek293T_directRNA_replicate5_run1.blow5_genome.eventalign.tsv"
OUT="/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/filtered/Hek293T_filtered1_eventalign.tsv.gz"

echo "Starting filtering"

awk -F'\t' '
BEGIN { OFS="\t" }

NR==1 {
  print
  next
}

# remove noisy events
$10=="NNNNN" { next }
$13=="inf"   { next }

# m1A candidate: center base of MODEL k-mer
substr($10,3,1)!="A" { next }

{
  print
}
' "$IN" | pigz -p 8 > "$OUT"

echo "[INFO] Finished at $(date)"
