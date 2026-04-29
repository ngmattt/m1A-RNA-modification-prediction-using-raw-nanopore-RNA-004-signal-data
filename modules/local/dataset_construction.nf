process FILTER_EVENTALIGN {
    tag "filter features"
    publishDir "${params.outdir}/03_filtered", mode: "copy"

    input:
    path eventalign

    output:
    path "HEK293T.filtered.eventalign.tsv.gz", emit: filtered

    script:
    """
    set -euo pipefail
    zcat "${eventalign}" | awk -F'\\t' '
    BEGIN { OFS="\\t" }
    NR==1 { print; next }
    \$10=="NNNNN" { next }
    \$13=="inf" { next }
    substr(\$10,3,1)!="A" { next }
    { print }
    ' | gzip -c > HEK293T.filtered.eventalign.tsv.gz
    """
}

process LABEL_EVENTALIGN {
    tag "label m1A"
    publishDir "${params.outdir}/04_labeled", mode: "copy"

    input:
    path filtered
    path bed

    output:
    path "HEK293T.filtered.labeled.tsv.gz", emit: labeled

    script:
    """
    set -euo pipefail
    python3 "${projectDir}/bin/label_eventalign_from_bed.py" \\
      --eventalign "${filtered}" \\
      --bed "${bed}" \\
      --output HEK293T.filtered.labeled.tsv.gz
    """
}

process DOWNSAMPLE_M1A {
    tag "site balance"
    publishDir "${params.outdir}/05_site_balanced", mode: "copy"

    input:
    path labeled

    output:
    path "m1A_site_balanced.tsv.gz", emit: site_balanced

    script:
    """
    set -euo pipefail
    python3 "${projectDir}/downsample_m1A.py" \\
      --input "${labeled}" \\
      --output m1A_site_balanced.tsv.gz \\
      --seed ${params.random_state}
    """
}

process BALANCE_DATASET {
    tag "row balance"
    publishDir "${params.outdir}/06_balanced_dataset", mode: "copy"

    input:
    path site_balanced

    output:
    path "m1A_fully_balanced.tsv.gz", emit: balanced

    script:
    """
    set -euo pipefail
    python3 "${projectDir}/bin/row_balance_eventalign_dataset.py" \\
      --input "${site_balanced}" \\
      --output m1A_fully_balanced.tsv.gz \\
      --random-state ${params.random_state}
    """
}
