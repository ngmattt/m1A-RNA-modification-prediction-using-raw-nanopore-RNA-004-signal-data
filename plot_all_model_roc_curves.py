#!/usr/bin/env python3

import os
from pathlib import Path

OUTPUT_DIR = Path(
    os.environ.get(
        "POSTER_ROC_OUTPUT_DIR",
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

import train_catboost as catboost_pipeline
import train_rf as rf_pipeline
import train_xgb as xgb_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPS = 1e-8

MODEL_COLORS = {
    "XGBoost": "#4C78A8",
    "Random Forest": "#F58518",
    "CatBoost": "#54A24B",
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


def resolve_path(env_var_name, local_filename, default_path):
    env_path = os.environ.get(env_var_name)
    if env_path:
        return Path(env_path)

    local_candidate = Path(local_filename)
    if local_candidate.exists():
        return local_candidate

    return Path(default_path)


def split_events_by_site(event_df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(event_df[["site"]], event_df["label"], groups=event_df["site"])
    )
    train_event_df = event_df.iloc[train_idx].reset_index(drop=True)
    test_event_df = event_df.iloc[test_idx].reset_index(drop=True)
    return train_event_df, test_event_df


def xgboost_predictions(raw_df):
    event_df, numeric_cols, categorical_cols = xgb_pipeline.prepare_event_table(raw_df)
    xgb_pipeline.validate_site_labels(event_df)
    train_event_df, test_event_df = split_events_by_site(event_df)

    train_site_df, feature_cols = xgb_pipeline.build_site_table(
        train_event_df,
        numeric_cols,
        categorical_cols,
    )
    test_site_df, _ = xgb_pipeline.build_site_table(
        test_event_df,
        numeric_cols,
        categorical_cols,
    )

    best_result = xgb_pipeline.select_best_model(train_site_df, feature_cols, "xgboost")

    final_models = []
    for model_id, result in enumerate(best_result["top_results"], start=1):
        model = xgb_pipeline.build_model("xgboost", result["params"], RANDOM_STATE + model_id)
        model.fit(train_site_df[feature_cols], train_site_df["label"])
        final_models.append(model)

    y_true = test_site_df["label"].to_numpy()
    y_prob = xgb_pipeline.ensemble_predict(final_models, test_site_df[feature_cols])
    return y_true, y_prob


def random_forest_predictions(raw_df):
    event_df, event_feature_cols = rf_pipeline.prepare_event_table(raw_df)
    rf_pipeline.validate_site_labels(event_df)
    train_event_df, test_event_df = split_events_by_site(event_df)

    train_site_df, site_feature_cols = rf_pipeline.build_site_table(train_event_df, event_feature_cols)
    test_site_df, _ = rf_pipeline.build_site_table(test_event_df, event_feature_cols)

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_site_df[site_feature_cols])
    x_test = imputer.transform(test_site_df[site_feature_cols])
    y_train = train_site_df["label"].to_numpy()
    y_true = test_site_df["label"].to_numpy()

    best_result = rf_pipeline.select_best_model(x_train, y_train)
    final_model = rf_pipeline.build_model(best_result["params"], RANDOM_STATE)
    final_model.fit(x_train, y_train)
    y_prob = final_model.predict_proba(x_test)[:, 1]
    return y_true, y_prob


def catboost_predictions(raw_df):
    event_df, numeric_cols, categorical_cols = xgb_pipeline.prepare_event_table(raw_df)
    xgb_pipeline.validate_site_labels(event_df)
    train_event_df, test_event_df = split_events_by_site(event_df)

    train_site_df, feature_cols = xgb_pipeline.build_site_table(
        train_event_df,
        numeric_cols,
        categorical_cols,
    )
    test_site_df, _ = xgb_pipeline.build_site_table(
        test_event_df,
        numeric_cols,
        categorical_cols,
    )

    x_train, x_test, cat_feature_indices = catboost_pipeline.prepare_catboost_tables(
        train_site_df,
        test_site_df,
        feature_cols,
        categorical_cols,
    )
    y_train = train_site_df["label"].to_numpy()
    y_true = test_site_df["label"].to_numpy()

    best_result = catboost_pipeline.select_best_model(x_train, y_train, cat_feature_indices)

    final_models = []
    for model_id, result in enumerate(best_result["top_results"], start=1):
        model = catboost_pipeline.build_model(result["params"], RANDOM_STATE + model_id)
        model.fit(x_train, y_train, cat_features=cat_feature_indices)
        final_models.append(model)

    y_prob = catboost_pipeline.ensemble_predict(final_models, x_test)
    return y_true, y_prob


def save_roc_plot(model_predictions):
    roc_rows = []
    metric_rows = []
    fig, ax = plt.subplots(figsize=(9.5, 8))

    for model_name, (y_true, y_prob) in model_predictions.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        metric_rows.append({"Model": model_name, "AUROC": auroc})

        for fpr_value, tpr_value, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append(
                {
                    "Model": model_name,
                    "False Positive Rate": fpr_value,
                    "True Positive Rate": tpr_value,
                    "Threshold": threshold,
                }
            )

        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model_name],
            linewidth=2.8,
            label=f"{model_name} (AUROC = {auroc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#8c8c8c",
        linewidth=1.8,
        label="Random classifier",
    )
    ax.set_title("ROC Curves for m1A Site Prediction Models", fontsize=19, pad=14)
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "all_models_roc_curves.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / "all_models_roc_curves.pdf", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUTPUT_DIR / "all_models_roc_curves.svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    pd.DataFrame(roc_rows).to_csv(OUTPUT_DIR / "all_models_roc_curve_points.tsv", sep="\t", index=False)
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUTPUT_DIR / "all_models_roc_auc.tsv", sep="\t", index=False)
    return metric_df


def main():
    data_path = resolve_path(
        "M1A_DATA_PATH",
        "m1A_fully_balanced.tsv.gz",
        "/Users/matthewng/Bioinformatics Spring 2026/rna_mod_project/m1A_fully_balanced.tsv.gz",
    )
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path, sep="\t", compression="gzip")

    print("\nReproducing original XGBoost pipeline...")
    xgb_true, xgb_prob = xgboost_predictions(raw_df)

    print("\nReproducing original Random Forest pipeline...")
    rf_true, rf_prob = random_forest_predictions(raw_df)

    print("\nReproducing original CatBoost pipeline...")
    cat_true, cat_prob = catboost_predictions(raw_df)

    model_predictions = {
        "XGBoost": (xgb_true, xgb_prob),
        "Random Forest": (rf_true, rf_prob),
        "CatBoost": (cat_true, cat_prob),
    }

    metric_df = save_roc_plot(model_predictions)
    print("\nAUROC values:")
    print(metric_df.to_string(index=False))
    print(f"\nSaved all-model ROC outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
