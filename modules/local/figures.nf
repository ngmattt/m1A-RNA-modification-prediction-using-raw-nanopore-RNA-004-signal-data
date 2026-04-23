process FIGURES {
    publishDir "${params.outdir}/07_figures", mode: "copy"

    input:
    path balanced

    output:
    path "figures/*", emit: out

    script:
    """
    set -euo pipefail
    mkdir -p figures
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    python3 "${projectDir}/generate_m1A_performance_figures.py"
    cp -R "${projectDir}/figures/." figures/ 2>/dev/null || true
    """
}
