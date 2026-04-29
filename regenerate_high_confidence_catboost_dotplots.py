#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_DIR = Path(
    os.environ.get(
        "M1A_ENRICHMENT_INPUT_DIR",
        "/Users/matthewng/Documents/New project/high_confidence_catboost_enrichment",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "M1A_ENRICHMENT_OUTPUT_DIR",
        str(INPUT_DIR),
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.family"] = "DejaVu Sans"


def save_dotplot(df, title, output_prefix):
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

    fig.savefig(OUTPUT_DIR / f"{output_prefix}.png", bbox_inches="tight", pad_inches=0.35, dpi=600)
    fig.savefig(OUTPUT_DIR / f"{output_prefix}.pdf", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(OUTPUT_DIR / f"{output_prefix}.svg", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main():
    tsv_files = sorted(INPUT_DIR.glob("*_high_confidence_catboost_enrichment.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No enrichment TSV files found in {INPUT_DIR}")

    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep="\t")
        library = tsv_path.name.replace("_high_confidence_catboost_enrichment.tsv", "")
        output_prefix = f"{library}_high_confidence_catboost_top_terms"
        title = f"High-Confidence CatBoost Enrichment: {library}"
        print(f"Plotting {library} from {tsv_path}")
        save_dotplot(df, title, output_prefix)

    print(f"\nSaved refreshed dot plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
