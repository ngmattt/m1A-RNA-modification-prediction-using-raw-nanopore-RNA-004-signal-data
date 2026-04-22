#!/usr/bin/env python3

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

import train_xgb as shared_pipeline


# ----------------------------
# Config
# ----------------------------
DEFAULT_DATA_PATH = (
    "/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data/"
    "m1A_fully_balanced.tsv.gz"
)
TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42
OUTPUT_PREFIX = "m1a_catboost_simple"
EPS = 1e-8


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
# Helper functions
# ----------------------------
def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, 0.0

    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + EPS)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + EPS)),
        "npv": float(tn / (tn + fn + EPS)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def catboost_param_grid():
    return [
        {
            "iterations": 300,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "subsample": 0.85,
        },
        {
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 5.0,
            "subsample": 0.80,
        },
        {
            "iterations": 700,
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 7.0,
            "subsample": 0.75,
        },
        {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 4.0,
            "subsample": 0.90,
        },
        {
            "iterations": 900,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 6.0,
            "subsample": 0.80,
        },
        {
            "iterations": 1200,
            "depth": 4,
            "learning_rate": 0.02,
            "l2_leaf_reg": 8.0,
            "subsample": 0.75,
        },
        {
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.04,
            "l2_leaf_reg": 3.0,
            "subsample": 0.85,
        },
        {
            "iterations": 1000,
            "depth": 3,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "subsample": 0.90,
        },
    ]


def build_model(params, random_state):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        bootstrap_type="Bernoulli",
        **params,
    )


def prepare_catboost_tables(train_site_df, test_site_df, feature_cols, categorical_cols):
    x_train = train_site_df[feature_cols].copy()
    x_test = test_site_df[feature_cols].copy()

    for col in categorical_cols:
        x_train[col] = x_train[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    cat_feature_indices = [x_train.columns.get_loc(col) for col in categorical_cols]
    return x_train, x_test, cat_feature_indices


def select_best_model(x_train, y_train, cat_feature_indices):
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_result = None
    ranked_results = []

    for index, params in enumerate(catboost_param_grid(), start=1):
        fold_prob = np.zeros(len(y_train), dtype=float)

        for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_train), start=1):
            model = build_model(params, RANDOM_STATE + fold_id)
            model.fit(
                x_train.iloc[fit_idx],
                y_train[fit_idx],
                cat_features=cat_feature_indices,
            )
            fold_prob[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]

        threshold, cv_f1 = best_f1_threshold(y_train, fold_prob)
        cv_auroc = roc_auc_score(y_train, fold_prob)
        cv_auprc = average_precision_score(y_train, fold_prob)
        cv_metrics = compute_metrics(y_train, fold_prob, threshold)

        result = {
            "params": params,
            "threshold": threshold,
            "cv_accuracy": float(cv_metrics["accuracy"]),
            "cv_precision": float(cv_metrics["precision"]),
            "cv_recall": float(cv_metrics["recall"]),
            "cv_f1": float(cv_f1),
            "cv_auroc": float(cv_auroc),
            "cv_auprc": float(cv_auprc),
        }
        ranked_results.append(result)

        if best_result is None or (
            result["cv_f1"],
            result["cv_auprc"],
            result["cv_auroc"],
        ) > (
            best_result["cv_f1"],
            best_result["cv_auprc"],
            best_result["cv_auroc"],
        ):
            best_result = result

        print(
            f"Candidate {index}: cv_accuracy={result['cv_accuracy']:.4f}, "
            f"cv_precision={result['cv_precision']:.4f}, "
            f"cv_recall={result['cv_recall']:.4f}, "
            f"cv_f1={result['cv_f1']:.4f}, "
            f"cv_auroc={result['cv_auroc']:.4f}, "
            f"cv_auprc={result['cv_auprc']:.4f}, "
            f"threshold={result['threshold']:.4f}"
        )

    ranked_results.sort(
        key=lambda row: (row["cv_f1"], row["cv_auprc"], row["cv_auroc"]),
        reverse=True,
    )
    best_result["top_results"] = [dict(row) for row in ranked_results[:3]]
    return best_result


def ensemble_predict(models, x_table):
    probabilities = [model.predict_proba(x_table)[:, 1] for model in models]
    return np.mean(np.vstack(probabilities), axis=0)


def print_metric_block(title, metrics):
    print(f"\n[{title}]")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"NPV:       {metrics['npv']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"AUPRC:     {metrics['auprc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

def main():
    data_path = resolve_data_path()
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path, sep="\t", compression="gzip")

    print("Building event-level features...")
    event_df, numeric_cols, categorical_cols = shared_pipeline.prepare_event_table(raw_df)
    shared_pipeline.validate_site_labels(event_df)

    print(f"Event rows: {len(event_df)}")
    print(f"Unique sites: {event_df['site'].nunique()}")
    print("Class distribution:")
    print(event_df["label"].value_counts().sort_index())

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
    train_site_df, feature_cols = shared_pipeline.build_site_table(
        train_event_df, numeric_cols, categorical_cols
    )
    test_site_df, _ = shared_pipeline.build_site_table(
        test_event_df, numeric_cols, categorical_cols
    )
    x_train, x_test, cat_feature_indices = prepare_catboost_tables(
        train_site_df, test_site_df, feature_cols, categorical_cols
    )
    y_train = train_site_df["label"].to_numpy()
    y_test = test_site_df["label"].to_numpy()

    print("\nTraining model: catboost")
    print(f"Train event rows: {len(train_event_df)}")
    print(f"Test event rows: {len(test_event_df)}")
    print(f"Train sites: {len(train_site_df)}")
    print(f"Test sites: {len(test_site_df)}")

    print("\nSelecting hyperparameters and threshold with site-level CV...")
    best_result = select_best_model(x_train, y_train, cat_feature_indices)

    print("\nBest grouped-CV selection:")
    print(best_result)

    final_models = []
    for model_id, result in enumerate(best_result["top_results"], start=1):
        model = build_model(result["params"], RANDOM_STATE + model_id)
        model.fit(x_train, y_train, cat_features=cat_feature_indices)
        final_models.append(model)

    site_prob = ensemble_predict(final_models, x_test)
    site_true = y_test
    site_metrics = compute_metrics(site_true, site_prob, best_result["threshold"])

    print_metric_block("SITE-LEVEL TEST METRICS", site_metrics)
    print("\nSite-level classification report:")
    print(
        classification_report(
            site_true,
            (site_prob >= best_result["threshold"]).astype(int),
            zero_division=0,
        )
    )

    artifact = {
        "models": final_models,
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "threshold": best_result["threshold"],
    }

    model_path = Path(f"{OUTPUT_PREFIX}.pkl")
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    print(f"\nSaved model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
