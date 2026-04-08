#!/bin/bash
#SBATCH --job-name=downsample_m1A
#SBATCH -p hopper
#SBATCH --qos hopper
#SBATCH -c 16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o downsample_m1A.%j.out
#SBATCH -e downsample_m1A.%j.err
#SBATCH -A r00270

set -eo pipefail

# Load conda

module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh

conda activate f5c_env

# Run python script

python /N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/labeled/downsample_m1A.py

echo "Job finished at $(date)"