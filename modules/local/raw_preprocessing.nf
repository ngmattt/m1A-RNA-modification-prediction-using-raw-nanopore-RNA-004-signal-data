process POD5_TO_BLOW5 {
    tag "HEK293T pod5->blow5"
    publishDir "${params.outdir}/00_blow5_conversion", mode: "copy"

    input:
    path pod5

    output:
    path "HEK293T.blow5", emit: blow5

    script:
    """
    set -euo pipefail
    blue-crab p2s "${pod5}" -o HEK293T.blow5 -c zlib -p 4 -t ${task.cpus}
    """
}

process DORADO_BASECALL {
    tag "HEK293T Dorado"
    publishDir "${params.outdir}/01_basecalling", mode: "copy"

    input:
    path pod5

    output:
    path "HEK293T.basecalled.bam", emit: bam
    path "HEK293T.fastq.gz", emit: fastq

    script:
    """
    set -euo pipefail
    dorado basecaller "${params.dorado_model}" "${pod5}" --emit-moves > HEK293T.basecalled.bam
    samtools fastq HEK293T.basecalled.bam | gzip > HEK293T.fastq.gz
    """
}

process ALIGN_FASTQ {
    tag "HEK293T alignment"
    publishDir "${params.outdir}/02_alignment", mode: "copy"

    input:
    path fastq
    path reference_fasta

    output:
    path "HEK293T.sorted.bam", emit: bam
    path "HEK293T.sorted.bam.bai", emit: bai

    script:
    """
    set -euo pipefail
    minimap2 -ax map-ont -t ${task.cpus} "${reference_fasta}" "${fastq}" | samtools sort -@ ${task.cpus} -o HEK293T.sorted.bam
    samtools index HEK293T.sorted.bam
    """
}

process F5C_EVENTALIGN {
    tag "HEK293T eventalign"
    publishDir "${params.outdir}/03_eventalign", mode: "copy"

    input:
    path fastq
    path bam
    path blow5
    path reference_fasta
    path kmer_model

    output:
    path "HEK293T.eventalign.tsv.gz", emit: eventalign

    script:
    """
    set -euo pipefail
    f5c index --slow5 "${blow5}" -t ${task.cpus} "${fastq}"
    f5c eventalign \\
      --rna \\
      --pore rna004 \\
      --reads "${fastq}" \\
      --bam "${bam}" \\
      --slow5 "${blow5}" \\
      --genome "${reference_fasta}" \\
      --kmer-model "${kmer_model}" \\
      --scale-events \\
      --signal-index \\
      --samples \\
      --min-mapq 0 \\
      --min-recalib-events 100 \\
      --skip-unreadable=yes \\
      --secondary=no \\
      -t ${task.cpus} \\
      -o HEK293T.eventalign.tsv
    gzip -f HEK293T.eventalign.tsv
    """
}
