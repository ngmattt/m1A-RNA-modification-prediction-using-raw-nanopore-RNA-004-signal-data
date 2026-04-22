process DORADO_BASECALL {
    tag "HEK293T Dorado"
    publishDir "${params.outdir}/00_basecalling", mode: "copy"

    input:
    path pod5

    output:
    path "HEK293T.fastq.gz", emit: fastq

    script:
    """
    set -euo pipefail
    dorado basecaller "${params.dorado_model}" "${pod5}" --emit-moves > HEK293T.bam
    samtools fastq HEK293T.bam | gzip > HEK293T.fastq.gz
    """
}

process ALIGN_FASTQ {
    tag "HEK293T alignment"
    publishDir "${params.outdir}/01_alignment", mode: "copy"

    input:
    path fastq
    path reference_fasta

    output:
    path "HEK293T.sorted.bam", emit: bam

    script:
    """
    set -euo pipefail
    minimap2 -ax map-ont "${reference_fasta}" "${fastq}" | samtools sort -o HEK293T.sorted.bam
    """
}

process F5C_EVENTALIGN {
    tag "HEK293T eventalign"
    publishDir "${params.outdir}/02_eventalign", mode: "copy"

    input:
    path fastq
    path bam
    path pod5
    path reference_fasta
    path kmer_model

    output:
    path "HEK293T.eventalign.tsv.gz", emit: eventalign

    script:
    """
    set -euo pipefail
    f5c index "${fastq}"
    f5c eventalign \\
      --rna \\
      --pore rna004 \\
      --reads "${fastq}" \\
      --bam "${bam}" \\
      --slow5 "${pod5}" \\
      --genome "${reference_fasta}" \\
      --kmer-model "${kmer_model}" \\
      --scale-events \\
      --signal-index \\
      --samples \\
      --min-mapq 0 \\
      --min-recalib-events 100 \\
      --skip-unreadable=yes \\
      --secondary=no \\
      -o HEK293T.eventalign.tsv
    gzip -f HEK293T.eventalign.tsv
    """
}
