process FIGURES {
    publishDir "${params.outdir}/09_figures", mode: "copy"

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

process POSTER_MODEL_PERFORMANCE {
    publishDir "${params.outdir}/10_poster_results/performance", mode: "copy"

    output:
    path "all_model_performance_figures/*", emit: out

    script:
    """
    set -euo pipefail
    export POSTER_PERFORMANCE_OUTPUT_DIR="all_model_performance_figures"
    python3 "${projectDir}/plot_all_model_performance.py"
    """
}

process POSTER_MODEL_ROC {
    publishDir "${params.outdir}/10_poster_results/roc", mode: "copy"

    input:
    path balanced

    output:
    path "all_model_performance_figures/*", emit: out

    script:
    """
    set -euo pipefail
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    export M1A_DATA_PATH="${PWD}/m1A_fully_balanced.tsv.gz"
    export POSTER_ROC_OUTPUT_DIR="all_model_performance_figures"
    python3 "${projectDir}/plot_all_model_roc_curves.py"
    """
}

process POSTER_MODEL_ACCURACY_LEARNING_CURVES {
    publishDir "${params.outdir}/10_poster_results/learning_curves", mode: "copy"

    input:
    path balanced

    output:
    path "model_learning_curves/*", emit: out

    script:
    """
    set -euo pipefail
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    export M1A_DATA_PATH="${PWD}/m1A_fully_balanced.tsv.gz"
    export LEARNING_CURVE_OUTPUT_DIR="model_learning_curves"
    python3 "${projectDir}/plot_model_accuracy_learning_curves.py"
    """
}

process HIGH_CONFIDENCE_ENRICHMENT {
    publishDir "${params.outdir}/10_poster_results/high_confidence_enrichment", mode: "copy"

    input:
    path balanced
    path bed

    output:
    path "high_confidence_catboost_enrichment/*", emit: out

    script:
    """
    set -euo pipefail
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    export M1A_DATA_PATH="${PWD}/m1A_fully_balanced.tsv.gz"
    export M1A_BED_PATH="${bed}"
    export M1A_OUTPUT_DIR="${PWD}/high_confidence_catboost_enrichment"
    python3 "${projectDir}/analyze_high_confidence_catboost_enrichment.py"
    """
}
