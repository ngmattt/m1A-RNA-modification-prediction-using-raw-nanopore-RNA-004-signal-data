#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { POD5_TO_BLOW5; DORADO_BASECALL; ALIGN_FASTQ; F5C_EVENTALIGN } from './modules/local/raw_preprocessing'
include { FILTER_EVENTALIGN; LABEL_EVENTALIGN; DOWNSAMPLE_M1A; BALANCE_DATASET } from './modules/local/dataset_construction'
include { TRAIN_XGB; TRAIN_RF; TRAIN_CATBOOST } from './modules/local/model_training'
include { FIGURES; POSTER_MODEL_PERFORMANCE; POSTER_MODEL_ROC; POSTER_MODEL_ACCURACY_LEARNING_CURVES; HIGH_CONFIDENCE_ENRICHMENT } from './modules/local/figures'

workflow {
    def balanced_ch
    def bed_for_enrichment = null

    if (params.raw_mode) {
        if (!params.hek293t_pod5 || !params.reference_fasta || !params.reference_kmer_model || !params.m1a_bed) {
            error "For --raw_mode true, provide --hek293t_pod5, --reference_fasta, --reference_kmer_model, and --m1a_bed"
        }

        pod5_ch = Channel.fromPath(params.hek293t_pod5, checkIfExists: true)
        ref_ch = Channel.fromPath(params.reference_fasta, checkIfExists: true)
        model_ch = Channel.fromPath(params.reference_kmer_model, checkIfExists: true)
        bed_ch = Channel.fromPath(params.m1a_bed, checkIfExists: true)

        blow5 = POD5_TO_BLOW5(pod5_ch)
        basecalled = DORADO_BASECALL(pod5_ch)
        aligned = ALIGN_FASTQ(basecalled.out.fastq, ref_ch)
        eventalign = F5C_EVENTALIGN(basecalled.out.fastq, aligned.out.bam, blow5.out.blow5, ref_ch, model_ch)
        filtered = FILTER_EVENTALIGN(eventalign.out.eventalign)
        labeled = LABEL_EVENTALIGN(filtered.out.filtered, bed_ch)
        site_balanced = DOWNSAMPLE_M1A(labeled.out.labeled)
        balanced = BALANCE_DATASET(site_balanced.out.site_balanced)
        balanced_ch = balanced.out.balanced
        bed_for_enrichment = bed_ch
    } else {
        if (!params.hek293t_labeled_events) {
            error "Provide --hek293t_labeled_events or enable --raw_mode true"
        }
        balanced_ch = Channel.fromPath(params.hek293t_labeled_events, checkIfExists: true)
        if (params.m1a_bed) {
            bed_for_enrichment = Channel.fromPath(params.m1a_bed, checkIfExists: true)
        }
    }

    TRAIN_XGB(balanced_ch)
    TRAIN_RF(balanced_ch)
    TRAIN_CATBOOST(balanced_ch)

    if (params.run_figures) {
        FIGURES(balanced_ch)
    }

    if (params.run_poster_results) {
        POSTER_MODEL_PERFORMANCE()
        POSTER_MODEL_ROC(balanced_ch)
        POSTER_MODEL_ACCURACY_LEARNING_CURVES(balanced_ch)

        if (params.run_high_confidence_enrichment) {
            if (bed_for_enrichment == null) {
                log.info "Skipping high-confidence enrichment because --m1a_bed was not provided."
            } else {
                HIGH_CONFIDENCE_ENRICHMENT(balanced_ch, bed_for_enrichment)
            }
        }
    }
}
