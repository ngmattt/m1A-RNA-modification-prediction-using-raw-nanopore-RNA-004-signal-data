# m1A RNA Modification Prediction Using Raw Nanopore RNA004 Signal Data

This repository contains a machine learning workflow for predicting **N1-methyladenosine (m1A)** RNA modification sites from **Oxford Nanopore direct RNA RNA004** signal-derived features. The project is organized around **site-level prediction**, where multiple nanopore events aligned to the same genomic or transcriptomic position are aggregated into one biologically meaningful prediction unit.

The repository now supports two ways of running the project:

- the original script-based workflow
- a new **Nextflow** workflow for portable end-to-end execution

## Project Overview

The main goal of this project is to test whether raw nanopore signal features contain enough information to distinguish **m1A-modified** from **unmodified** sites.

The workflow starts from `f5c eventalign`-style signal tables and can also be run from raw nanopore signal input when using Nextflow. Candidate sites are labeled with a HEK293T BED file, balanced at both the site and row levels, and then modeled with tree-based classifiers.

Models compared in this project:

- **XGBoost**
- **Random Forest**
- **CatBoost**

## Dataset

The balanced development dataset contains nanopore event-level rows with columns such as:

- `contig`
- `position`
- `reference_kmer`
- `read_index`
- `strand`
- `event_index`
- `event_level_mean`
- `event_stdv`
- `event_length`
- `model_kmer`
- `model_mean`
- `model_stdv`
- `standardized_level`
- `start_idx`
- `end_idx`
- `samples`
- `label`

A site is defined as:

```text
site = contig:position
```

The dataset used for model development contains:

- **53,622 event rows**
- **1,032 unique sites**
- **26,811 negative** event rows
- **26,811 positive** event rows

## Modeling Strategy

### 1. Event-level feature engineering

Event-level features include:

- observed vs expected signal differences
- signal ratios
- event span and normalized event length
- summary statistics from the raw `samples` vector
- k-mer and sequence-context features

### 2. Site-level aggregation

Because m1A is interpreted at the site level, event features are grouped by `contig:position` and summarized across reads using:

- mean
- standard deviation
- minimum
- maximum
- median
- quantiles

### 3. Site-aware splitting

Train/test splitting is performed by **site**, not by individual event rows, to reduce read-level leakage.

### 4. Model selection

Hyperparameters are selected using grouped cross-validation, and final performance is measured on held-out sites.

## Repository Layout

### Core model scripts

- `train_xgb.py`
- `train_rf.py`
- `train_catboost.py`
- `generate_m1A_performance_figures.py`

These are the primary entry points for model development and evaluation. The repository now uses these filenames directly rather than alias wrapper scripts.

### Original preprocessing and labeling scripts

- `dorado_basecall.sh`
- `fastq_to_bam.sh`
- `f5c_index_eventalign.sh`
- `filter_f5c_features.sh`
- `label_m1A_filtered.py`
- `downsample_m1A.py`
- `m1A_row_balance.sh`

### Nextflow workflow

- `main.nf`
- `nextflow.config`
- `NEXTFLOW.md`
- `modules/local/`
- `bin/`

### Nextflow helper scripts

- `bin/label_eventalign_from_bed.py`
- `bin/row_balance_eventalign_dataset.py`

## Nextflow Workflow

The Nextflow workflow packages the repository into a more portable pipeline.

### Supported modes

#### 1. Start from an already balanced HEK293T table

```bash
nextflow run main.nf \
  --hek293t_labeled_events /path/to/m1A_fully_balanced.tsv.gz
```

#### 2. Start from raw HEK293T nanopore signal data

```bash
nextflow run main.nf -profile conda \
  --raw_mode true \
  --hek293t_pod5 /path/to/HEK293T_run.pod5 \
  --reference_fasta /path/to/reference.fa \
  --reference_kmer_model /path/to/rna004.nucleotide.5mer.model \
  --m1a_bed /path/to/HEK293T_m1A_sites.bed
```

### Raw-mode steps

1. POD5 to BLOW5 conversion with `convert_pod5_to_blow5.sh` logic
2. Dorado basecalling to BAM with `dorado_basecall.sh` logic
3. FASTQ extraction from the Dorado BAM
4. Alignment with `fastq_to_bam.sh` logic
5. `f5c index` and `f5c eventalign` with `f5c_index_eventalign.sh` logic
6. Eventalign filtering
7. BED-based labeling
8. Site balancing with `downsample_m1A.py`
9. Row balancing
10. XGBoost training
11. Random Forest training
12. CatBoost training
13. Performance figure generation
14. Poster results figure generation

### Main Nextflow outputs

```text
results/00_blow5_conversion/
results/01_basecalling/
results/02_alignment/
results/03_eventalign/
results/04_filtered/
results/05_labeled/
results/06_site_balanced/
results/07_balanced_dataset/
results/08_models/
results/09_figures/
results/10_poster_results/
```

See [NEXTFLOW.md](NEXTFLOW.md) for more detail.

## Script-Based Usage

The model scripts can still be run directly:

```bash
python3 train_xgb.py
python3 train_rf.py
python3 train_catboost.py
python3 generate_m1A_performance_figures.py
```

These scripts now first look for `m1A_fully_balanced.tsv.gz` in the current working directory. If it is not present, they fall back to the original hard-coded HPC path used during development.

## Dependencies

Python 3.10+ is recommended.

Main Python packages used:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `catboost`
- `matplotlib`
- `seaborn`
- `gseapy`

For the Nextflow pipeline, common command-line tools include:

- `nextflow`
- `dorado`
- `minimap2`
- `samtools`
- `f5c`

An example conda environment for the Nextflow workflow is provided in `environment.yml`.

## Performance Summary

Held-out site-level performance:

| Model | Accuracy | Precision | Recall | F1-score | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.8599 | 0.8235 | 0.8842 | 0.8528 | 0.9494 | 0.9472 |
| Random Forest | 0.8599 | 0.9459 | 0.7368 | 0.8284 | 0.9092 | 0.9246 |
| CatBoost | 0.8841 | 0.8817 | 0.8632 | 0.8723 | 0.9514 | 0.9503 |

**CatBoost** achieved the strongest overall held-out performance across accuracy, F1-score, AUROC, and AUPRC, while **Random Forest** produced the highest precision.

## Figures

The repository includes figure generation for:

- ROC curves
- precision-recall curves
- confusion matrices
- model metric comparison plots
- all-model poster metric plots
- all-model poster ROC curves
- model accuracy learning curves
- high-confidence CatBoost enrichment plots

Example outputs:

- `figures/m1a_roc_pr_curves.png`
- `figures/m1a_roc_pr_curves.svg`
- `figures/m1a_confusion_matrices.png`
- `figures/m1a_confusion_matrices.svg`
- `figures/m1a_metric_comparison.png`
- `figures/m1a_metric_comparison.svg`
- `all_model_performance_figures/all_model_metric_comparison.svg`
- `all_model_performance_figures/all_models_roc_curves.svg`
- `model_learning_curves/model_accuracy_learning_curves.svg`
- `high_confidence_catboost_enrichment/KEGG_2026_high_confidence_catboost_top_terms.svg`

PNG figures are exported at 600 dpi, and SVG files are also written for vector-based manuscript or poster editing.

For the poster-specific results section, the main scripts are:

- `plot_all_model_performance.py`
- `plot_all_model_roc_curves.py`
- `plot_model_accuracy_learning_curves.py`
- `analyze_high_confidence_catboost_enrichment.py`

## Notes

- The project is designed around **site-level m1A prediction**.
- The balanced dataset assumes each site has a consistent label.
- The Nextflow implementation follows the repository workflow while removing hard-coded HPC-specific assumptions where possible.
- The Nextflow dataset construction now explicitly includes the repository's original `downsample_m1A.py` site-balancing step before row balancing.
- Some original shell scripts were written for a specific cluster environment, so the Nextflow pipeline reproduces their logic in a more portable structure through `modules/local/` and `bin/`.

## Future Directions

Possible extensions include:

- prediction on unlabeled HeLa candidate sites
- independent external validation
- calibration analysis
- SHAP-based interpretation
- transcriptome-wide testing on naturally imbalanced candidate sets
- extension to other RNA modifications

## Citation

If you use this repository, please cite the associated project materials and any downstream manuscript or capstone outputs once available.
