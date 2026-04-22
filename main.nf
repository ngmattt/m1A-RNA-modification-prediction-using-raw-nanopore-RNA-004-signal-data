#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { DORADO_BASECALL; ALIGN_FASTQ; F5C_EVENTALIGN } from './modules/local/raw_preprocessing'
include { FILTER_EVENTALIGN; LABEL_EVENTALIGN; BALANCE_DATASET } from './modules/local/dataset_construction'
include { TRAIN_XGB; TRAIN_RF; TRAIN_CATBOOST } from './modules/local/model_training'
include { FIGURES } from './modules/local/figures'

workflow {
    def balanced_ch

    if (params.raw_mode) {
        if (!params.hek293t_pod5 || !params.reference_fasta || !params.reference_kmer_model || !params.m1a_bed) {
            error "For --raw_mode true, provide --hek293t_pod5, --reference_fasta, --reference_kmer_model, and --m1a_bed"
        }

        pod5_ch = Channel.fromPath(params.hek293t_pod5, checkIfExists: true)
        ref_ch = Channel.fromPath(params.reference_fasta, checkIfExists: true)
        model_ch = Channel.fromPath(params.reference_kmer_model, checkIfExists: true)
        bed_ch = Channel.fromPath(params.m1a_bed, checkIfExists: true)

        basecalled = DORADO_BASECALL(pod5_ch)
        aligned = ALIGN_FASTQ(basecalled.out.fastq, ref_ch)
        eventalign = F5C_EVENTALIGN(basecalled.out.fastq, aligned.out.bam, pod5_ch, ref_ch, model_ch)
        filtered = FILTER_EVENTALIGN(eventalign.out.eventalign)
        labeled = LABEL_EVENTALIGN(filtered.out.filtered, bed_ch)
        balanced = BALANCE_DATASET(labeled.out.labeled)
        balanced_ch = balanced.out.balanced
    } else {
        if (!params.hek293t_labeled_events) {
            error "Provide --hek293t_labeled_events or enable --raw_mode true"
        }
        balanced_ch = Channel.fromPath(params.hek293t_labeled_events, checkIfExists: true)
    }

    TRAIN_XGB(balanced_ch)
    TRAIN_RF(balanced_ch)
    TRAIN_CATBOOST(balanced_ch)

    if (params.run_figures) {
        FIGURES(balanced_ch)
    }
}
