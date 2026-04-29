#!/usr/bin/env python3

import os
from pathlib import Path

WORKDIR = Path.cwd()
OUTPUT_DIR = WORKDIR / "figures"
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

import train_xgb as xgb_script
import train_rf as rf_script


# ----------------------------
# Config
# ----------------------------
DEFAULT_DATA_PATH = (
    "/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data/"
    "m1A_fully_balanced.tsv.gz"
)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def resolve_data_path():
    candidates = [
        Path("m1A_fully_balanced.tsv.gz"),
        Path.cwd() / "m1A_fully_balanced.tsv.gz",
    ]

    env_path = Path(os.environ["M1A_DATA_PATH"]) if "M1A_DATA_PATH" in os.environ else None
    if env_path is not None:
        candidates.insert(0, env_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(DEFAULT_DATA_PATH)


# ----------------------------
# Plot settings
# ----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.family"] = "DejaVu Sans"

SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 600,
}


def compute_summary_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred),
    }


def split_raw_data(raw_df):
    site_ids = raw_df["contig"].astype(str) + ":" + raw_df["position"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(raw_df[["contig"]], raw_df["label"], groups=site_ids)
    )
    return (
        raw_df.iloc[train_idx].reset_index(drop=True),
        raw_df.iloc[test_idx].reset_index(drop=True),
    )


def train_and_score_xgboost(train_raw_df, test_raw_df):
    train_event_df, numeric_cols, categorical_cols = xgb_script.prepare_event_table(train_raw_df)
    test_event_df, _, _ = xgb_script.prepare_event_table(test_raw_df)

    train_site_df, feature_cols = xgb_script.build_site_table(
        train_event_df, numeric_cols, categorical_cols
    )
    test_site_df, _ = xgb_script.build_site_table(
        test_event_df, numeric_cols, categorical_cols
    )

    model_name = xgb_script.choose_model_name(xgb_script.MODEL_CHOICE)
    best_result = xgb_script.select_best_model(train_site_df, feature_cols, model_name)

    final_models = []
    for model_id, result in enumerate(best_result["top_results"], start=1):
        model = xgb_script.build_model(
            model_name,
            result["params"],
            xgb_script.RANDOM_STATE + model_id,
        )
        model.fit(train_site_df[feature_cols], train_site_df["label"])
        final_models.append(model)

    y_true = test_site_df["label"].to_numpy()
    y_prob = xgb_script.ensemble_predict(final_models, test_site_df[feature_cols])
    return y_true, y_prob, best_result["threshold"]


def train_and_score_random_forest(train_raw_df, test_raw_df):
    train_event_df, event_feature_cols = rf_script.prepare_event_table(train_raw_df)
    test_event_df, _ = rf_script.prepare_event_table(test_raw_df)

    train_site_df, site_feature_cols = rf_script.build_site_table(
        train_event_df, event_feature_cols
    )
    test_site_df, _ = rf_script.build_site_table(
        test_event_df, event_feature_cols
    )

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_site_df[site_feature_cols])
    x_test = imputer.transform(test_site_df[site_feature_cols])
    y_train = train_site_df["label"].to_numpy()
    y_true = test_site_df["label"].to_numpy()

    best_result = rf_script.select_best_model(x_train, y_train)
    final_model = rf_script.build_model(best_result["params"], rf_script.RANDOM_STATE)
    final_model.fit(x_train, y_train)
    y_prob = final_model.predict_proba(x_test)[:, 1]
    return y_true, y_prob, best_result["threshold"]


def save_roc_pr_figure(xgb_true, xgb_prob, rf_true, rf_prob):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    xgb_fpr, xgb_tpr, _ = roc_curve(xgb_true, xgb_prob)
    rf_fpr, rf_tpr, _ = roc_curve(rf_true, rf_prob)
    axes[0].plot(
        xgb_fpr,
        xgb_tpr,
        linewidth=2.5,
        color="#1f77b4",
        label=f"XGBoost (AUROC = {roc_auc_score(xgb_true, xgb_prob):.3f})",
    )
    axes[0].plot(
        rf_fpr,
        rf_tpr,
        linewidth=2.5,
        color="#d62728",
        label=f"Random Forest (AUROC = {roc_auc_score(rf_true, rf_prob):.3f})",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=True, fontsize=10, loc="lower right")

    xgb_precision, xgb_recall, _ = precision_recall_curve(xgb_true, xgb_prob)
    rf_precision, rf_recall, _ = precision_recall_curve(rf_true, rf_prob)
    axes[1].plot(
        xgb_recall,
        xgb_precision,
        linewidth=2.5,
        color="#1f77b4",
        label=f"XGBoost (AUPRC = {average_precision_score(xgb_true, xgb_prob):.3f})",
    )
    axes[1].plot(
        rf_recall,
        rf_precision,
        linewidth=2.5,
        color="#d62728",
        label=f"Random Forest (AUPRC = {average_precision_score(rf_true, rf_prob):.3f})",
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=True, fontsize=10, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "m1a_roc_pr_curves.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / "m1a_roc_pr_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "m1a_roc_pr_curves.svg", bbox_inches="tight")
    plt.close(fig)


def save_confusion_figure(xgb_metrics, rf_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    sns.heatmap(
        np.array(xgb_metrics["ConfusionMatrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("XGBoost Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        np.array(rf_metrics["ConfusionMatrix"]),
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Random Forest Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "m1a_confusion_matrices.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / "m1a_confusion_matrices.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "m1a_confusion_matrices.svg", bbox_inches="tight")
    plt.close(fig)


def save_metrics_barplot(xgb_metrics, rf_metrics):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "AUROC", "AUPRC"]
    plot_df = pd.DataFrame(
        {
            "Metric": metric_names * 2,
            "Score": [xgb_metrics[name] for name in metric_names]
            + [rf_metrics[name] for name in metric_names],
            "Model": ["XGBoost"] * len(metric_names)
            + ["Random Forest"] * len(metric_names),
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#1f77b4", "#d62728"],
        ax=ax,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Model Performance on Held-Out m1A Sites")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(frameon=True)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "m1a_metric_comparison.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / "m1a_metric_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "m1a_metric_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading dataset...")
    data_path = resolve_data_path()
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path, sep="\t", compression="gzip")

    print("Rebuilding the same held-out site split...")
    train_raw_df, test_raw_df = split_raw_data(raw_df)

    print("Training and scoring XGBoost model...")
    xgb_true, xgb_prob, xgb_threshold = train_and_score_xgboost(train_raw_df, test_raw_df)
    xgb_metrics = compute_summary_metrics(xgb_true, xgb_prob, xgb_threshold)

    print("Training and scoring Random Forest model...")
    rf_true, rf_prob, rf_threshold = train_and_score_random_forest(train_raw_df, test_raw_df)
    rf_metrics = compute_summary_metrics(rf_true, rf_prob, rf_threshold)

    print("\nXGBoost metrics:")
    for key, value in xgb_metrics.items():
        if key != "ConfusionMatrix":
            print(f"{key}: {value:.4f}")
    print(xgb_metrics["ConfusionMatrix"])

    print("\nRandom Forest metrics:")
    for key, value in rf_metrics.items():
        if key != "ConfusionMatrix":
            print(f"{key}: {value:.4f}")
    print(rf_metrics["ConfusionMatrix"])

    print("\nSaving figures...")
    save_roc_pr_figure(xgb_true, xgb_prob, rf_true, rf_prob)
    save_confusion_figure(xgb_metrics, rf_metrics)
    save_metrics_barplot(xgb_metrics, rf_metrics)
    print(f"Figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
