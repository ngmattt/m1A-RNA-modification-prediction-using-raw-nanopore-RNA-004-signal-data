# Nextflow Workflow

This repository now includes a Nextflow workflow for automating the m1A nanopore pipeline.

## Structure

The workflow is organized into:

- `main.nf`: top-level orchestration
- `modules/local/raw_preprocessing.nf`: POD5 to BLOW5 conversion, Dorado basecalling, alignment, and `f5c index/eventalign`
- `modules/local/dataset_construction.nf`: filtering, BED labeling, site downsampling, and row balancing
- `modules/local/model_training.nf`: XGBoost, Random Forest, and CatBoost training
- `modules/local/figures.nf`: performance figure generation plus poster results figures
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

1. POD5 to BLOW5 conversion
2. Dorado basecalling to BAM
3. FASTQ extraction from BAM
4. Alignment with minimap2 to create the BAM used by `f5c`
5. `f5c index`
6. `f5c eventalign`
7. Eventalign filtering
8. BED-based labeling
9. Site balancing with `downsample_m1A.py`
10. Row balancing
11. XGBoost training
12. Random Forest training
13. CatBoost training
14. Performance figure generation
15. Poster results figure generation

## Main outputs

```text
results/00_blow5_conversion/
results/01_basecalling/
results/02_alignment/
results/03_eventalign/
results/04_filtered/
results/05_labeled/
results/06_site_balanced/m1A_site_balanced.tsv.gz
results/07_balanced_dataset/m1A_fully_balanced.tsv.gz
results/08_models/xgb/
results/08_models/rf/
results/08_models/catboost/
results/09_figures/
results/10_poster_results/performance/
results/10_poster_results/roc/
results/10_poster_results/learning_curves/
results/10_poster_results/high_confidence_enrichment/
```

## Notes

- This Nextflow implementation follows the repository scripts and reproduces the project workflow in a portable way.
- In raw mode, the pipeline now explicitly follows the repository methodology: `POD5 -> BLOW5`, `POD5 -> Dorado BAM -> FASTQ`, `FASTQ -> aligned BAM`, then `f5c index/eventalign`.
- In raw mode, BED labels are used to construct the balanced HEK293T training dataset automatically.
- The helper scripts in `bin/` replace brittle inline code and make the workflow easier to test independently.
- Poster results generation is controlled by `--run_poster_results true|false`.
- High-confidence CatBoost enrichment is wired into the pipeline but defaults to off because Enrichr access may require outbound network access. Enable it with `--run_high_confidence_enrichment true` and provide `--m1a_bed`.
- HeLa inference can be added as a downstream extension after the core repository workflow is stabilized.
