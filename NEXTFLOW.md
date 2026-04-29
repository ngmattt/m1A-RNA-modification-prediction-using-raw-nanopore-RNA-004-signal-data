# Nextflow Workflow

This repository now includes a Nextflow workflow for automating the m1A nanopore pipeline.

## Structure

The workflow is organized into:

- `main.nf`: top-level orchestration
- `modules/local/raw_preprocessing.nf`: Dorado, alignment, and `f5c eventalign`
- `modules/local/dataset_construction.nf`: filtering, BED labeling, site downsampling, and row balancing
- `modules/local/model_training.nf`: XGBoost, Random Forest, and CatBoost training
- `modules/local/figures.nf`: performance figure generation
- `bin/label_eventalign_from_bed.py`: BED-based labeling helper
- `downsample_m1A.py`: site-balancing step from the original repository workflow
- `bin/row_balance_eventalign_dataset.py`: row-balancing helper

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
7. Site balancing with `downsample_m1A.py`
8. Row balancing
9. XGBoost training
10. Random Forest training
11. CatBoost training
12. Performance figure generation

## Main outputs

```text
results/05_site_balanced/m1A_site_balanced.tsv.gz
results/06_balanced_dataset/m1A_fully_balanced.tsv.gz
results/07_models/xgb/
results/07_models/rf/
results/07_models/catboost/
results/08_figures/
```

## Notes

- This Nextflow implementation follows the repository scripts and reproduces the project workflow in a portable way.
- In raw mode, BED labels are used to construct the balanced HEK293T training dataset automatically.
- The helper scripts in `bin/` replace brittle inline code and make the workflow easier to test independently.
- HeLa inference can be added as a downstream extension after the core repository workflow is stabilized.
