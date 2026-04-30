#!/usr/bin/env python3

import os
from pathlib import Path

OUTPUT_DIR = Path(
    os.environ.get(
        "LEARNING_CURVE_OUTPUT_DIR",
        "model_learning_curves",
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
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

import train_rf as rf_pipeline
import train_xgb as xgb_pipeline

RANDOM_STATE = 42
CV_FOLDS = 5
TRAIN_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.family"] = "DejaVu Sans"

SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "pad_inches": 0.25,
    "dpi": 600,
}

score_palette = {
    "Training accuracy": "#2F6F73",
    "CV accuracy": "#D95F02",
}


def resolve_path(env_var_name, local_filename, default_path):
    env_path = os.environ.get(env_var_name)
    if env_path:
        return Path(env_path)

    local_candidate = Path(local_filename)
    if local_candidate.exists():
        return local_candidate

    return Path(default_path)


def make_xgboost_model():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=180,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.60,
        min_child_weight=10,
        reg_lambda=20.0,
        reg_alpha=5.0,
    )


def make_random_forest_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        min_samples_split=16,
        max_features=0.40,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_catboost_model():
    return CatBoostClassifier(
        iterations=250,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=20.0,
        min_data_in_leaf=10,
        subsample=0.85,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        bootstrap_type="Bernoulli",
    )


def build_xgb_catboost_site_table(raw_df):
    event_df, numeric_cols, categorical_cols = xgb_pipeline.prepare_event_table(raw_df)
    site_df, feature_cols = xgb_pipeline.build_site_table(event_df, numeric_cols, categorical_cols)
    return site_df.sort_values("site").reset_index(drop=True), feature_cols, categorical_cols


def build_random_forest_site_table(raw_df, ordered_sites):
    event_df, event_feature_cols = rf_pipeline.prepare_event_table(raw_df)
    site_df, feature_cols = rf_pipeline.build_site_table(event_df, event_feature_cols)
    site_df = site_df.set_index("site").loc[ordered_sites].reset_index()
    return site_df, feature_cols


def remove_coordinate_and_identifier_features(feature_cols, categorical_cols=None):
    blocked_exact = {"contig", "site_event_index_span"}
    blocked_prefixes = ("position_", "read_index_", "event_index_", "start_idx_", "end_idx_")

    filtered_features = [
        col
        for col in feature_cols
        if col not in blocked_exact
        and not any(col.startswith(prefix) for prefix in blocked_prefixes)
    ]

    if categorical_cols is None:
        return filtered_features

    filtered_categorical = [col for col in categorical_cols if col in filtered_features]
    return filtered_features, filtered_categorical


def stratified_subset_indices(y, fraction, seed):
    if fraction >= 1.0:
        return np.arange(len(y))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=fraction,
        random_state=seed,
    )
    subset_idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return subset_idx


def add_accuracy_rows(results, model_name, fold, fraction, train_size, y_train, y_train_pred, y_cv, y_cv_pred):
    results.append(
        {
            "Model": model_name,
            "Fold": fold,
            "Training Fraction": fraction,
            "Training Sites Used": train_size,
            "Score Type": "Training accuracy",
            "Accuracy": accuracy_score(y_train, y_train_pred),
        }
    )
    results.append(
        {
            "Model": model_name,
            "Fold": fold,
            "Training Fraction": fraction,
            "Training Sites Used": train_size,
            "Score Type": "CV accuracy",
            "Accuracy": accuracy_score(y_cv, y_cv_pred),
        }
    )


def run_xgboost_curve(site_df, feature_cols, cv_splits):
    results = []

    for fold, (train_idx, cv_idx) in enumerate(cv_splits, start=1):
        train_df = site_df.iloc[train_idx].reset_index(drop=True)
        cv_df = site_df.iloc[cv_idx].reset_index(drop=True)
        y_train_full = train_df["label"].to_numpy()

        for fraction in TRAIN_FRACTIONS:
            subset_idx = stratified_subset_indices(y_train_full, fraction, RANDOM_STATE + fold)
            train_subset = train_df.iloc[subset_idx]

            model = make_xgboost_model()
            model.fit(train_subset[feature_cols], train_subset["label"])

            train_pred = model.predict(train_subset[feature_cols])
            cv_pred = model.predict(cv_df[feature_cols])
            add_accuracy_rows(
                results,
                "XGBoost",
                fold,
                fraction,
                len(train_subset),
                train_subset["label"],
                train_pred,
                cv_df["label"],
                cv_pred,
            )

    return results


def run_random_forest_curve(site_df, feature_cols, cv_splits):
    results = []

    for fold, (train_idx, cv_idx) in enumerate(cv_splits, start=1):
        train_df = site_df.iloc[train_idx].reset_index(drop=True)
        cv_df = site_df.iloc[cv_idx].reset_index(drop=True)
        y_train_full = train_df["label"].to_numpy()

        for fraction in TRAIN_FRACTIONS:
            subset_idx = stratified_subset_indices(y_train_full, fraction, RANDOM_STATE + fold)
            train_subset = train_df.iloc[subset_idx]

            imputer = SimpleImputer(strategy="median")
            train_x = imputer.fit_transform(train_subset[feature_cols])
            cv_x = imputer.transform(cv_df[feature_cols])

            model = make_random_forest_model()
            model.fit(train_x, train_subset["label"])

            train_pred = model.predict(train_x)
            cv_pred = model.predict(cv_x)
            add_accuracy_rows(
                results,
                "Random Forest",
                fold,
                fraction,
                len(train_subset),
                train_subset["label"],
                train_pred,
                cv_df["label"],
                cv_pred,
            )

    return results


def run_catboost_curve(site_df, feature_cols, categorical_cols, cv_splits):
    results = []
    cat_site_df = site_df.copy()

    for col in categorical_cols:
        cat_site_df[col] = cat_site_df[col].astype(str)

    cat_feature_indices = [
        cat_site_df[feature_cols].columns.get_loc(col)
        for col in categorical_cols
    ]

    for fold, (train_idx, cv_idx) in enumerate(cv_splits, start=1):
        train_df = cat_site_df.iloc[train_idx].reset_index(drop=True)
        cv_df = cat_site_df.iloc[cv_idx].reset_index(drop=True)
        y_train_full = train_df["label"].to_numpy()

        for fraction in TRAIN_FRACTIONS:
            subset_idx = stratified_subset_indices(y_train_full, fraction, RANDOM_STATE + fold)
            train_subset = train_df.iloc[subset_idx]

            model = make_catboost_model()
            model.fit(
                train_subset[feature_cols],
                train_subset["label"],
                cat_features=cat_feature_indices,
            )

            train_pred = model.predict(train_subset[feature_cols]).astype(int)
            cv_pred = model.predict(cv_df[feature_cols]).astype(int)
            add_accuracy_rows(
                results,
                "CatBoost",
                fold,
                fraction,
                len(train_subset),
                train_subset["label"],
                train_pred,
                cv_df["label"],
                cv_pred,
            )

    return results


def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", **SAVEFIG_KWARGS)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.25)


def main():
    data_path = resolve_path(
        "M1A_DATA_PATH",
        "m1A_fully_balanced.tsv.gz",
        "/Users/matthewng/Bioinformatics Spring 2026/rna_mod_project/m1A_fully_balanced.tsv.gz",
    )
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path, sep="\t", compression="gzip")

    print("Building site-level feature tables...")
    xgb_cat_site_df, xgb_cat_features, categorical_cols = build_xgb_catboost_site_table(raw_df)
    rf_site_df, rf_features = build_random_forest_site_table(raw_df, xgb_cat_site_df["site"].tolist())
    xgb_cat_features, categorical_cols = remove_coordinate_and_identifier_features(
        xgb_cat_features,
        categorical_cols,
    )
    rf_features = remove_coordinate_and_identifier_features(rf_features)

    print("Using regularized models and excluding coordinate/identifier-like features.")
    print(f"XGBoost/CatBoost features: {len(xgb_cat_features)}")
    print(f"Random Forest features: {len(rf_features)}")

    if not np.array_equal(xgb_cat_site_df["label"].to_numpy(), rf_site_df["label"].to_numpy()):
        raise ValueError("XGBoost/CatBoost and Random Forest site labels are not aligned.")

    y = xgb_cat_site_df["label"].to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(np.zeros(len(y)), y))

    print("Computing XGBoost accuracy learning curve...")
    all_results = run_xgboost_curve(xgb_cat_site_df, xgb_cat_features, cv_splits)

    print("Computing Random Forest accuracy learning curve...")
    all_results.extend(run_random_forest_curve(rf_site_df, rf_features, cv_splits))

    print("Computing CatBoost accuracy learning curve...")
    all_results.extend(run_catboost_curve(xgb_cat_site_df, xgb_cat_features, categorical_cols, cv_splits))

    results_df = pd.DataFrame(all_results)
    metrics_path = OUTPUT_DIR / "model_accuracy_learning_curve_metrics.tsv"
    results_df.to_csv(metrics_path, sep="\t", index=False)

    model_order = ["XGBoost", "Random Forest", "CatBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, model_name in zip(axes, model_order):
        model_df = results_df[results_df["Model"] == model_name]
        sns.lineplot(
            data=model_df,
            x="Training Sites Used",
            y="Accuracy",
            hue="Score Type",
            hue_order=["Training accuracy", "CV accuracy"],
            palette=score_palette,
            marker="o",
            errorbar="sd",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Training Sites Used")
        ax.set_ylabel("Accuracy" if ax is axes[0] else "")
        ax.set_ylim(0.45, 1.03)
        ax.legend(title="", frameon=True, loc="lower right")

    fig.suptitle("Accuracy Learning Curves by Model", fontsize=22, y=1.03)
    fig.tight_layout()
    save_figure(fig, "model_accuracy_learning_curves")
    plt.close(fig)

    cv_only = results_df[results_df["Score Type"] == "CV accuracy"]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.lineplot(
        data=cv_only,
        x="Training Sites Used",
        y="Accuracy",
        hue="Model",
        hue_order=model_order,
        marker="o",
        errorbar="sd",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Cross-Validation Accuracy vs Training Set Size")
    ax.set_xlabel("Training Sites Used")
    ax.set_ylabel("CV Accuracy")
    ax.set_ylim(0.45, 1.03)
    ax.legend(title="", frameon=True, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "model_cv_accuracy_learning_curves")
    plt.close(fig)

    summary_df = (
        results_df.groupby(["Model", "Training Fraction", "Training Sites Used", "Score Type"])["Accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = OUTPUT_DIR / "model_accuracy_learning_curve_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print(f"Saved accuracy learning curves to: {OUTPUT_DIR}")
    print(f"Saved fold-level metrics to: {metrics_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()
