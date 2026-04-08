#!/bin/bash
#SBATCH -J count_m1A_labels
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o count_m1A_labels.%j.out
#SBATCH -e count_m1A_labels.%j.err
#SBATCH -A r00270

set -eo pipefail

FILE="/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/labeled/Hek293T_filtered_labeled_eventalign1.tsv.gz"

zcat "$FILE" | awk -F'\t' 'NR>1 {c[$NF]++} END {print "0:", c[0]; print "1:", c[1]}'
zcat "$FILE" | awk -F'\t' 'NR>1 {print $1":"$2":"$NF}' | sort | uniq | wc -l

# positive sites
zcat "$FILE" | awk -F'\t' 'NR>1 && $NF==1 {print $1":"$2}' | sort | uniq | wc -l

# negative sites
zcat "$FILE" | awk -F'\t' 'NR>1 && $NF==0 {print $1":"$2}' | sort | uniq | wc -l

echo " [INFO] Finished $(date)" 
