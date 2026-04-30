#!/usr/bin/env python3

import os
from pathlib import Path

OUTPUT_DIR = Path(
    os.environ.get(
        "POSTER_PERFORMANCE_OUTPUT_DIR",
        "all_model_performance_figures",
    )
)
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


metrics = [
    {
        "Model": "XGBoost",
        "Accuracy": 0.8599,
        "Balanced Accuracy": 0.8617,
        "Precision": 0.8235,
        "Recall": 0.8842,
        "F1-score": 0.8528,
        "MCC": 0.7211,
        "AUROC": 0.9494,
        "AUPRC": 0.9472,
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.8599,
        "Balanced Accuracy": 0.8506,
        "Precision": 0.9459,
        "Recall": 0.7368,
        "F1-score": 0.8284,
        "MCC": 0.7290,
        "AUROC": 0.9092,
        "AUPRC": 0.9246,
    },
    {
        "Model": "CatBoost",
        "Accuracy": 0.8841,
        "Balanced Accuracy": 0.8825,
        "Precision": 0.8817,
        "Recall": 0.8632,
        "F1-score": 0.8723,
        "MCC": 0.7663,
        "AUROC": 0.9514,
        "AUPRC": 0.9503,
    },
]

confusion_matrices = {
    "XGBoost": np.array([[94, 18], [11, 84]]),
    "Random Forest": np.array([[108, 4], [25, 70]]),
    "CatBoost": np.array([[101, 11], [13, 82]]),
}


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.family"] = "DejaVu Sans"

SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "pad_inches": 0.25,
    "dpi": 600,
}

palette = {
    "XGBoost": "#4C78A8",
    "Random Forest": "#F58518",
    "CatBoost": "#54A24B",
}


def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.25)


def main():
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "all_model_site_level_metrics.tsv", sep="\t", index=False)

    main_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUROC", "AUPRC"]
    main_plot_df = metrics_df.melt(
        id_vars="Model",
        value_vars=main_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=main_plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Held-Out Site-Level Performance Across Models")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="", loc="lower right", frameon=True)

    for container in ax.containers:
        labels = ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        for label in labels:
            label.set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14)
    save_figure(fig, "all_model_metric_comparison")
    plt.close(fig)

    heatmap_df = metrics_df.set_index("Model")[
        ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "MCC", "AUROC", "AUPRC"]
    ]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.70,
        vmax=1.00,
        linewidths=0.5,
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title("Extended Site-Level Performance Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.24)
    save_figure(fig, "all_model_metric_heatmap")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (model_name, matrix) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            square=True,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticklabels(["Unmodified", "m1A"], rotation=0)
        ax.set_yticklabels(["Unmodified", "m1A"], rotation=0)

    fig.suptitle("Confusion Matrices on Held-Out Sites", y=1.03)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.16, wspace=0.35)
    save_figure(fig, "all_model_confusion_matrices")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _, row in metrics_df.iterrows():
        ax.scatter(
            row["Recall"],
            row["Precision"],
            s=260,
            color=palette[row["Model"]],
            label=row["Model"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            row["Recall"] + 0.006,
            row["Precision"] + 0.006,
            row["Model"],
            fontsize=10,
        )

    ax.set_title("Precision-Recall Tradeoff by Model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.70, 0.92)
    ax.set_ylim(0.80, 0.98)
    ax.legend(title="", frameon=True, loc="lower left")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.13)
    save_figure(fig, "all_model_precision_recall_tradeoff")
    plt.close(fig)

    print(f"Saved all model performance figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
