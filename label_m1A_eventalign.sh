#!/bin/bash
#SBATCH -J m1A_genomic_label
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o m1A_genomic_label.%j.out
#SBATCH -e m1A_genomic_label.%j.err
#SBATCH -A r00270

set -eo pipefail

module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
conda activate f5c_env

label_script="/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/labeled/label_m1A_filtered.py"
python "$label_script"

