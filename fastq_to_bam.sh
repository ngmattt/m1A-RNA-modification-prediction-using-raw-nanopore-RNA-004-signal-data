#!/bin/bash 
#SBATCH -J fastq_to_bam
#SBATCH -o fastq_to_bam_%A_%a.out
#SBATCH -e fastq_to_bam_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngmat@iu.edu
#SBATCH -A r00270

set -euo pipefail

module load conda/25.3.0
source /geode2/soft/hps/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
conda activate f5c_env

minimap2 -ax map-ont -t 16   /N/project/NGS-JangaLab/Matthew/rna_seq_data/reference/gencode.v49.transcripts.fa  SGNex_Hek293T_directRNA_replicate5_run1.fastq.gz | samtools sort -o HEK293T_replicate5_transcriptome.bam
