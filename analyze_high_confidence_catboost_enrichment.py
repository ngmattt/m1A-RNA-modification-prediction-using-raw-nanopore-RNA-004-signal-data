#!/usr/bin/env python3

import os
from pathlib import Path

OUTPUT_DIR = Path("/N/project/NGS-JangaLab/Matthew/ML_data/figures/high_confidence_catboost_enrichment")
OUTPUT_DIR.mkdir(exist_ok=True)

mpl_dir = OUTPUT_DIR / ".mpl-cache"
font_dir = OUTPUT_DIR / ".font-cache"
mpl_dir.mkdir(exist_ok=True)
font_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)
os.environ["XDG_CACHE_HOME"] = str(font_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from sklearn.model_selection import StratifiedKFold

import train_m1a_model as shared_pipeline
import train_m1a_catboost_simple as catboost_pipeline


# ----------------------------
# Config
# ----------------------------
DATA_PATH = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data/m1A_fully_balanced.tsv.gz"
BED_PATH = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/raw_data/HEK293T_RNA004/rna_mods/HEK293T_m1A_sites.bed"
HIGH_CONFIDENCE_THRESHOLD = 0.80
MIN_GENE_COUNT = 30
LIBRARIES = [
    "GO_Biological_Process_2025",
    "GO_Molecular_Function_2025",
    "GO_Cellular_Component_2025",
    "KEGG_2026",
    "Reactome_Pathways_2024",
]


# ----------------------------
# Plot settings
# ----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"


def save_dotplot(df, title, filename):
    top_df = df.head(12).copy()
    fig, ax = plt.subplots(figsize=(17, 9.5))
    sns.scatterplot(
        data=top_df,
        x="-log10(FDR)",
        y="Term",
        size="Combined Score",
        hue="Odds Ratio",
        palette="viridis",
        sizes=(70, 260),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("-log10 Adjusted P-value")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.46, right=0.78, top=0.90, bottom=0.12)
    fig.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def save_barplot(df, title, filename):
    top_df = df.head(12).copy()
    fig, ax = plt.subplots(figsize=(16, 8.5))
    sns.barplot(data=top_df, x="-log10(FDR)", y="Term", color="#1f77b4", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("-log10 Adjusted P-value")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.90, bottom=0.12)
    fig.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def build_bed_annotation_table():
    bed_df = pd.read_csv(BED_PATH, sep="\t", header=None)
    bed_df = bed_df.rename(columns={0: "chrom", 1: "start", 15: "gene_names", 16: "gene_biotypes"})
    bed_df["site"] = bed_df["chrom"].astype(str) + ":" + bed_df["start"].astype(str)
    return bed_df


def extract_protein_coding_genes(annotation_df):
    genes = set()
    for _, row in annotation_df.iterrows():
        gene_names = [x.strip() for x in str(row["gene_names"]).split(",")]
        biotypes = [x.strip() for x in str(row["gene_biotypes"]).split(",")]
        for gene, biotype in zip(gene_names, biotypes):
            if not gene or gene.lower() in {"na", "intergenic"}:
                continue
            if biotype != "protein_coding":
                continue
            if gene.startswith("(") and gene.endswith(")"):
                continue
            genes.add(gene)
    return sorted(genes)


def oof_probabilities(x_table, y, cat_feature_indices, params):
    splitter = StratifiedKFold(
        n_splits=catboost_pipeline.CV_FOLDS,
        shuffle=True,
        random_state=catboost_pipeline.RANDOM_STATE,
    )
    probs = np.zeros(len(y), dtype=float)
    for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_table, y), start=1):
        model = catboost_pipeline.build_model(params, catboost_pipeline.RANDOM_STATE + fold_id)
        model.fit(
            x_table.iloc[fit_idx],
            y[fit_idx],
            cat_features=cat_feature_indices,
        )
        probs[val_idx] = model.predict_proba(x_table.iloc[val_idx])[:, 1]
    return probs


print("Loading labeled event dataset...")
raw_df = pd.read_csv(DATA_PATH, sep="\t", compression="gzip")

print("Building site-level feature table...")
event_df, numeric_cols, categorical_cols = shared_pipeline.prepare_event_table(raw_df)
site_df, feature_cols = shared_pipeline.build_site_table(event_df, numeric_cols, categorical_cols)

x_table = site_df[feature_cols].copy()
for col in categorical_cols:
    x_table[col] = x_table[col].astype(str)
y = site_df["label"].to_numpy()
cat_feature_indices = [x_table.columns.get_loc(col) for col in categorical_cols]

print("Selecting best CatBoost configuration on all labeled sites...")
best_result = catboost_pipeline.select_best_model(x_table, y, cat_feature_indices)
print(best_result)

print("Generating out-of-fold probabilities for the best CatBoost model...")
oof_prob = oof_probabilities(x_table, y, cat_feature_indices, best_result["params"])
site_df["oof_probability"] = oof_prob
site_df["high_confidence_positive"] = site_df["oof_probability"] >= HIGH_CONFIDENCE_THRESHOLD

selected_df = site_df.loc[site_df["high_confidence_positive"]].copy()
if selected_df["label"].sum() < MIN_GENE_COUNT:
    fallback_threshold = 0.70
    print(
        f"Only {int(selected_df['label'].sum())} labeled positives at probability >= {HIGH_CONFIDENCE_THRESHOLD:.2f}. "
        f"Falling back to {fallback_threshold:.2f}."
    )
    selected_df = site_df.loc[site_df["oof_probability"] >= fallback_threshold].copy()
    selected_threshold = fallback_threshold
else:
    selected_threshold = HIGH_CONFIDENCE_THRESHOLD

selected_df.to_csv(OUTPUT_DIR / "high_confidence_catboost_sites.tsv", sep="\t", index=False)

print(f"Selected sites at probability >= {selected_threshold:.2f}: {len(selected_df)}")
print(selected_df[["site", "label", "oof_probability"]].head(20).to_string(index=False))

print("Matching high-confidence predicted sites to BED annotations...")
bed_df = build_bed_annotation_table()
annotated_df = selected_df.merge(bed_df, on="site", how="inner")
annotated_df.to_csv(OUTPUT_DIR / "high_confidence_catboost_sites_annotated.tsv", sep="\t", index=False)

print(f"Annotated high-confidence sites matched to BED: {len(annotated_df)}")

gene_list = extract_protein_coding_genes(annotated_df)
pd.DataFrame({"gene_symbol": gene_list}).to_csv(
    OUTPUT_DIR / "high_confidence_catboost_protein_coding_genes.tsv",
    sep="\t",
    index=False,
)

print(f"Protein-coding genes from high-confidence CatBoost predictions: {len(gene_list)}")
print(gene_list[:40])

all_results = []
for library in LIBRARIES:
    print(f"\nRunning enrichment for {library} ...")
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=library,
        organism="human",
        outdir=None,
        no_plot=True,
    )

    result_df = enr.results.copy()
    if result_df.empty:
        continue

    result_df["Library"] = library
    result_df["-log10(FDR)"] = -np.log10(result_df["Adjusted P-value"].clip(lower=1e-300))
    result_df = result_df.sort_values("Adjusted P-value", ascending=True)
    result_df.to_csv(
        OUTPUT_DIR / f"{library}_high_confidence_catboost_enrichment.tsv",
        sep="\t",
        index=False,
    )
    all_results.append(result_df)
    print(result_df[["Term", "Adjusted P-value", "Odds Ratio", "Combined Score"]].head(10).to_string(index=False))

    if library.startswith("GO_") or library.startswith("KEGG_"):
        save_dotplot(
            result_df,
            f"High-Confidence CatBoost Enrichment: {library}",
            f"{library}_high_confidence_catboost_top_terms",
        )
    else:
        save_barplot(
            result_df,
            f"High-Confidence CatBoost Enrichment: {library}",
            f"{library}_high_confidence_catboost_top_terms",
        )

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "combined_high_confidence_catboost_enrichment.tsv", sep="\t", index=False)

print(f"\nOutputs saved to: {OUTPUT_DIR}")
