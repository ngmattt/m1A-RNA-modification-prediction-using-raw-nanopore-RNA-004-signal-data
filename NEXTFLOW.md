# Nextflow Workflow

This repository now includes a Nextflow workflow for automating the m1A nanopore pipeline.

## Structure

The workflow is organized into:

- `main.nf`: top-level orchestration
- `modules/local/raw_preprocessing.nf`: Dorado, alignment, and `f5c eventalign`
- `modules/local/dataset_construction.nf`: filtering, BED labeling, and balancing
- `modules/local/model_training.nf`: XGBoost, Random Forest, and CatBoost training
- `modules/local/figures.nf`: performance figure generation
- `bin/label_eventalign_from_bed.py`: BED-based labeling helper
- `bin/balance_eventalign_dataset.py`: site and row balancing helper

## Supported modes

### 1. Start from a prebuilt balanced HEK293T table

```bash
nextflow run main.nf \
  --hek293t_labeled_events /path/to/m1A_fully_balanced.tsv.gz
```

### 2. Start from raw HEK293T nanopore signal data

```bash
nextflow run main.nf -profile conda \
  --raw_mode true \
  --hek293t_pod5 /path/to/HEK293T_run.pod5 \
  --reference_fasta /path/to/reference.fa \
  --reference_kmer_model /path/to/rna004.nucleotide.5mer.model \
  --m1a_bed /path/to/HEK293T_m1A_sites.bed
```

## Pipeline steps in raw mode

1. Dorado basecalling
2. FASTQ extraction
3. Alignment with minimap2
4. `f5c eventalign`
5. Eventalign filtering
6. BED-based labeling
7. Site balancing and row balancing
8. XGBoost training
9. Random Forest training
10. CatBoost training
11. Performance figure generation

## Main outputs

```text
results/05_balanced_dataset/m1A_fully_balanced.tsv.gz
results/06_models/xgb/
results/06_models/rf/
results/06_models/catboost/
results/07_figures/
```

## Notes

- This Nextflow implementation follows the repository scripts and reproduces the project workflow in a portable way.
- In raw mode, BED labels are used to construct the balanced HEK293T training dataset automatically.
- The helper scripts in `bin/` replace brittle inline code and make the workflow easier to test independently.
- HeLa inference can be added as a downstream extension after the core repository workflow is stabilized.
