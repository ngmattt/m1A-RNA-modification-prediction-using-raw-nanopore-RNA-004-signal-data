#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


# ----------------------------
# Config
# ----------------------------
DATA_PATH = (
    "/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data/"
    "m1A_fully_balanced.tsv.gz"
)
TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42
OUTPUT_PREFIX = "m1a_random_forest_simple"
EPS = 1e-8


# ----------------------------
# Helper functions
# ----------------------------
def safe_divide(a, b):
    return a / (b + EPS)


def hamming_distance(a, b):
    left = "" if pd.isna(a) else str(a)
    right = "" if pd.isna(b) else str(b)
    width = max(len(left), len(right))
    left = left.ljust(width, "N")
    right = right.ljust(width, "N")
    return sum(x != y for x, y in zip(left, right))


def empty_sample_summary():
    return {
        "sample_count": 0,
        "sample_mean": np.nan,
        "sample_std": np.nan,
        "sample_min": np.nan,
        "sample_max": np.nan,
        "sample_median": np.nan,
        "sample_q10": np.nan,
        "sample_q25": np.nan,
        "sample_q75": np.nan,
        "sample_q90": np.nan,
        "sample_iqr": np.nan,
        "sample_range": np.nan,
        "sample_first": np.nan,
        "sample_last": np.nan,
        "sample_slope": np.nan,
        "sample_mean_abs_step": np.nan,
    }


def summarize_samples(sample_string):
    if pd.isna(sample_string) or not str(sample_string).strip():
        return empty_sample_summary()

    values = np.fromstring(str(sample_string), sep=",", dtype=float)
    if values.size == 0:
        return empty_sample_summary()

    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    if values.size > 1:
        x_axis = np.arange(values.size, dtype=float)
        slope = np.polyfit(x_axis, values, 1)[0]
        mean_abs_step = np.abs(np.diff(values)).mean()
    else:
        slope = 0.0
        mean_abs_step = 0.0

    return {
        "sample_count": int(values.size),
        "sample_mean": float(values.mean()),
        "sample_std": float(values.std(ddof=0)),
        "sample_min": float(values.min()),
        "sample_max": float(values.max()),
        "sample_median": float(np.median(values)),
        "sample_q10": float(q10),
        "sample_q25": float(q25),
        "sample_q75": float(q75),
        "sample_q90": float(q90),
        "sample_iqr": float(q75 - q25),
        "sample_range": float(values.max() - values.min()),
        "sample_first": float(values[0]),
        "sample_last": float(values[-1]),
        "sample_slope": float(slope),
        "sample_mean_abs_step": float(mean_abs_step),
    }


def add_event_features(df):
    df = df.copy()
    df["site"] = df["contig"].astype(str) + ":" + df["position"].astype(str)

    sample_table = df["samples"].apply(summarize_samples).apply(pd.Series)
    df = pd.concat([df, sample_table], axis=1)

    df["signal_diff"] = df["event_level_mean"] - df["model_mean"]
    df["abs_diff"] = df["signal_diff"].abs()
    df["signal_ratio"] = safe_divide(df["event_level_mean"], df["model_mean"])
    df["z_signal_diff"] = safe_divide(df["signal_diff"], df["model_stdv"])
    df["event_span"] = df["end_idx"] - df["start_idx"]
    df["length_per_sample"] = safe_divide(df["event_length"], df["sample_count"])
    df["sample_mean_minus_model"] = df["sample_mean"] - df["model_mean"]
    df["sample_std_ratio"] = safe_divide(df["sample_std"], df["event_stdv"])
    df["sample_range_ratio"] = safe_divide(df["sample_range"], df["event_stdv"])
    df["kmer_match"] = (df["reference_kmer"] == df["model_kmer"]).astype(int)
    df["kmer_hamming"] = [
        hamming_distance(a, b)
        for a, b in zip(df["reference_kmer"], df["model_kmer"])
    ]
    df["is_mito"] = df["contig"].astype(str).eq("chrM").astype(int)
    df["reference_gc_count"] = df["reference_kmer"].astype(str).apply(
        lambda text: sum(base in {"G", "C"} for base in text)
    )
    df["model_gc_count"] = df["model_kmer"].astype(str).apply(
        lambda text: sum(base in {"G", "C"} for base in text)
    )

    for index in range(5):
        ref_col = f"reference_base_{index}"
        model_col = f"model_base_{index}"
        df[ref_col] = df["reference_kmer"].astype(str).str[index]
        df[model_col] = df["model_kmer"].astype(str).str[index]
        for base in "ACGTN":
            df[f"{ref_col}_{base}"] = (df[ref_col] == base).astype(int)
            df[f"{model_col}_{base}"] = (df[model_col] == base).astype(int)

    return df


def get_event_feature_columns():
    numeric_cols = [
        "position", "read_index", "event_index", "event_level_mean", "event_stdv",
        "event_length", "model_mean", "model_stdv", "standardized_level",
        "start_idx", "end_idx", "sample_count", "sample_mean", "sample_std",
        "sample_min", "sample_max", "sample_median", "sample_q10", "sample_q25",
        "sample_q75", "sample_q90", "sample_iqr", "sample_range", "sample_first",
        "sample_last", "sample_slope", "sample_mean_abs_step", "signal_diff",
        "abs_diff", "signal_ratio", "z_signal_diff", "event_span",
        "length_per_sample", "sample_mean_minus_model", "sample_std_ratio",
        "sample_range_ratio", "kmer_match", "kmer_hamming", "is_mito",
        "reference_gc_count", "model_gc_count",
    ]

    one_hot_cols = []
    for index in range(5):
        for base in "ACGTN":
            one_hot_cols.append(f"reference_base_{index}_{base}")
            one_hot_cols.append(f"model_base_{index}_{base}")

    return numeric_cols, one_hot_cols


def prepare_event_table(df):
    df = add_event_features(df)
    numeric_cols, one_hot_cols = get_event_feature_columns()

    for col in numeric_cols + one_hot_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, numeric_cols + one_hot_cols


def build_site_table(event_df, event_feature_cols):
    numeric_df = event_df[event_feature_cols].copy()
    numeric_df["site"] = event_df["site"].values
    grouped = numeric_df.groupby("site")

    site_df = pd.concat(
        [
            grouped.mean().add_suffix("_mean"),
            grouped.std().fillna(0.0).add_suffix("_std"),
            grouped.min().add_suffix("_min"),
            grouped.max().add_suffix("_max"),
            grouped.median().add_suffix("_median"),
        ],
        axis=1,
    )
    site_df["site_read_count"] = grouped.size().astype(float)
    site_df["site_signal_diff_range"] = (
        grouped["signal_diff"].max() - grouped["signal_diff"].min()
    ).astype(float)
    site_df["site_abs_diff_mean"] = grouped["abs_diff"].mean().astype(float)

    site_labels = event_df.groupby("site").agg(label=("label", "max"))
    site_df = site_df.join(site_labels, how="inner").reset_index()

    feature_cols = [col for col in site_df.columns if col not in {"site", "label"}]
    return site_df, feature_cols


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


def rf_param_grid():
    return [
        {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 700,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
            "max_features": 0.35,
        },
        {
            "n_estimators": 900,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 600,
            "max_depth": 20,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": 0.45,
        },
    ]


def build_model(params, random_state):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def select_best_model(x_train, y_train):
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    best_result = None

    for index, params in enumerate(rf_param_grid(), start=1):
        fold_prob = np.zeros(len(y_train), dtype=float)

        for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_train), start=1):
            model = build_model(params, RANDOM_STATE + fold_id)
            model.fit(x_train[fit_idx], y_train[fit_idx])
            fold_prob[val_idx] = model.predict_proba(x_train[val_idx])[:, 1]

        threshold, _ = best_f1_threshold(y_train, fold_prob)
        metrics = compute_metrics(y_train, fold_prob, threshold)

        result = {
            "params": params,
            "threshold": threshold,
            "cv_accuracy": metrics["accuracy"],
            "cv_precision": metrics["precision"],
            "cv_recall": metrics["recall"],
            "cv_f1": metrics["f1"],
            "cv_auroc": metrics["auroc"],
            "cv_auprc": metrics["auprc"],
        }

        if best_result is None or (
            result["cv_auprc"],
            result["cv_f1"],
            result["cv_auroc"],
        ) > (
            best_result["cv_auprc"],
            best_result["cv_f1"],
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

    return best_result


def validate_site_labels(event_df):
    mixed_sites = int((event_df.groupby("site")["label"].nunique() > 1).sum())
    if mixed_sites:
        raise ValueError(f"Found {mixed_sites} sites with mixed labels.")


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


# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset...")
raw_df = pd.read_csv(DATA_PATH, sep="\t", compression="gzip")


# ----------------------------
# Build event-level features
# ----------------------------
print("Building event features...")
event_df, event_feature_cols = prepare_event_table(raw_df)
validate_site_labels(event_df)


# ----------------------------
# Split by site
# ----------------------------
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
train_site_df, site_feature_cols = build_site_table(train_event_df, event_feature_cols)
test_site_df, _ = build_site_table(test_event_df, event_feature_cols)

print(f"Event rows: {len(event_df)}")
print(f"Train sites: {len(train_site_df)}")
print(f"Test sites: {len(test_site_df)}")
print("Class distribution:")
print(train_site_df["label"].value_counts().sort_index())


# ----------------------------
# Prepare matrices
# ----------------------------
imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(train_site_df[site_feature_cols])
x_test = imputer.transform(test_site_df[site_feature_cols])
y_train = train_site_df["label"].to_numpy()
y_test = test_site_df["label"].to_numpy()


# ----------------------------
# Model selection
# ----------------------------
print("\nSelecting Random Forest hyperparameters...")
best_result = select_best_model(x_train, y_train)

print("\nBest CV configuration:")
print(best_result)


# ----------------------------
# Train final model
# ----------------------------
final_model = build_model(best_result["params"], RANDOM_STATE)
final_model.fit(x_train, y_train)


# ----------------------------
# Test set predictions
# ----------------------------
test_prob = final_model.predict_proba(x_test)[:, 1]
test_metrics = compute_metrics(y_test, test_prob, best_result["threshold"])

print_metric_block("SITE-LEVEL TEST METRICS", test_metrics)
print("\nClassification report:")
print(
    classification_report(
        y_test,
        (test_prob >= best_result["threshold"]).astype(int),
        zero_division=0,
    )
)


# ----------------------------
# Save outputs
# ----------------------------
artifact = {
    "model": final_model,
    "imputer": imputer,
    "site_feature_columns": site_feature_cols,
    "event_feature_columns": event_feature_cols,
    "threshold": best_result["threshold"],
}

model_path = Path(f"{OUTPUT_PREFIX}.joblib")
dump(artifact, model_path)

print(f"\nSaved model to: {model_path.resolve()}")
