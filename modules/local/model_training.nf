process TRAIN_XGB {
    publishDir "${params.outdir}/08_models/xgb", mode: "copy"

    input:
    path balanced

    output:
    path "xgb/*", emit: out

    script:
    """
    set -euo pipefail
    mkdir -p xgb
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    python3 "${projectDir}/train_xgb.py"
    mv m1a_site_model_simple* xgb/ 2>/dev/null || true
    """
}

process TRAIN_RF {
    publishDir "${params.outdir}/08_models/rf", mode: "copy"

    input:
    path balanced

    output:
    path "rf/*", emit: out

    script:
    """
    set -euo pipefail
    mkdir -p rf
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    python3 "${projectDir}/train_rf.py"
    mv m1a_random_forest_simple* rf/ 2>/dev/null || true
    """
}

process TRAIN_CATBOOST {
    publishDir "${params.outdir}/08_models/catboost", mode: "copy"

    input:
    path balanced

    output:
    path "catboost/*", emit: out

    script:
    """
    set -euo pipefail
    mkdir -p catboost
    cp "${balanced}" m1A_fully_balanced.tsv.gz
    python3 "${projectDir}/train_catboost.py"
    mv m1a_catboost_simple* catboost/ 2>/dev/null || true
    """
}
