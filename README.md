# m1A RNA Modification Prediction Using Raw Nanopore RNA004 Signal Data

This project uses raw Oxford Nanopore direct RNA sequencing signal-derived features to predict **N1-methyladenosine (m1A)** modification sites. The workflow is designed around **site-level prediction**, where multiple nanopore events from the same genomic site are summarized and used to classify whether that site is modified.

The project currently includes:

- an **XGBoost-based classifier** for site-level m1A prediction
- a **Random Forest baseline**
- scripts for **model training and evaluation**
- a script to generate **publication-style performance figures**

## Project Goal

The main goal of this project is to determine whether raw signal features from Nanopore **RNA004** direct RNA sequencing contain enough information to distinguish **m1A-modified** from **unmodified** sites.

Instead of predicting modification status for individual events only, this project focuses on **site-level classification**, which is more biologically meaningful because RNA modification is typically interpreted at the transcriptomic position level.

## Dataset

The training data is a tab-separated gzip-compressed file of nanopore event-level measurements:

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

Each row represents one nanopore event. A site is defined as:

```text
site = contig:position
```

The dataset used in development contains:

- **53,622 event rows**
- **1,032 unique sites**
- balanced event labels: **26,811 negative** and **26,811 positive**

## Modeling Strategy

### 1. Event-level feature engineering

For each event, the scripts derive additional signal features such as:

- signal difference between observed and expected current
- signal ratios
- event span and normalized event length
- summary statistics from the raw `samples` vector
- sequence-context features from `reference_kmer` and `model_kmer`

### 2. Site-level aggregation

Because the prediction target is the **modification site**, not a single event, event features are grouped by site and summarized using:

- mean
- standard deviation
- minimum
- maximum
- median
- quantiles

This reduces noise across reads and creates one feature vector per site.

### 3. Site-aware train/test splitting

To avoid leakage, train/test splitting is performed by **site**, not by row. This ensures that events from the same site do not appear in both training and test sets.

### 4. Model selection

The project compares:

- **XGBoost**
- **Random Forest**

Hyperparameters are selected using cross-validation on the training sites, and final performance is measured on held-out test sites.

## Repository Files

### Main model scripts

- [`train_m1a_model_simple.py`](/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/train_m1a_model_simple.py)  
  Simplified XGBoost-based site-level classifier script.

- [`train_m1a_random_forest_simple.py`](/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/train_m1a_random_forest_simple.py)  
  Simplified Random Forest site-level classifier script.

### Figure generation

- [`generate_m1a_performance_figures.py`](/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/generate_m1a_performance_figures.py)  
  Generates ROC curves, precision-recall curves, confusion matrices, and metric comparison plots for both models using `matplotlib` and `seaborn`.

### Additional training scripts

- [`train_m1a_model.py`](/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/train_m1a_model.py)
- [`train_m1a_random_forest.py`](/N/project/NGS-JangaLab/Matthew/rna_seq_data/scripts/train_m1a_random_forest.py)

These contain the same modeling ideas in a more modular form.

## Requirements

Python 3.10+ is recommended.

Main Python packages used:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

Install dependencies with:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## How To Run

### Train the XGBoost model

```bash
python3 train_m1a_model_simple.py
```

### Train the Random Forest model

```bash
python3 train_m1a_random_forest_simple.py
```

### Generate performance figures

```bash
python3 generate_m1a_performance_figures.py
```

Generated figures are saved in:

```text
figures/
```

## Performance Summary

### XGBoost

Held-out site-level test performance:

- Accuracy: **0.8599**
- Precision: **0.8235**
- Recall: **0.8842**
- F1-score: **0.8528**
- AUROC: **0.9494**
- AUPRC: **0.9472**

### Random Forest

Held-out site-level test performance:

- Accuracy: **0.8599**
- Precision: **0.9459**
- Recall: **0.7368**
- F1-score: **0.8284**
- AUROC: **0.9092**
- AUPRC: **0.9246**

### Interpretation

The **XGBoost model** provides the best overall balance for this task, especially in:

- recall
- F1-score
- AUROC
- AUPRC

The **Random Forest model** achieves higher precision, but misses more true modified sites.

## Figures

The project includes traditional performance visualizations for manuscript preparation:

- ROC curve comparison
- precision-recall curve comparison
- confusion matrices
- bar plot comparing Accuracy, Precision, Recall, F1-score, AUROC, and AUPRC

Example output files:

- `figures/m1a_roc_pr_curves.png`
- `figures/m1a_confusion_matrices.png`
- `figures/m1a_metric_comparison.png`

## Notes

- The current workflow assumes that each site has a **consistent label**.
- The scripts are written for **site-level m1A prediction**, which is more suitable for biological interpretation than per-event classification alone.
- The figure script retrains both models directly from the model scripts rather than loading serialized model objects.

## Future Directions

Possible extensions include:

- prediction on unlabeled candidate sites
- feature importance analysis
- SHAP-based model interpretation
- calibration analysis
- external validation on independent nanopore datasets
- comparison with deep learning approaches

## Citation

If you use this repository in a manuscript, thesis, or presentation, please cite the repository and the associated study once available.
