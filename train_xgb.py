#!/usr/bin/env python3

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


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
MODEL_CHOICE = "auto"   # "auto", "xgboost", or "histgb"
OUTPUT_PREFIX = "m1a_site_model_simple"
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
        "sample_skew": np.nan,
    }


def summarize_samples(sample_string):
    if pd.isna(sample_string) or not str(sample_string).strip():
        return empty_sample_summary()

    values = np.fromstring(str(sample_string), sep=",", dtype=float)
    if values.size == 0:
        return empty_sample_summary()

    mean = values.mean()
    std = values.std(ddof=0)
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    centered = values - mean
    skew = np.mean(centered ** 3) / ((std ** 3) + EPS)

    if values.size > 1:
        x_axis = np.arange(values.size, dtype=float)
        slope = np.polyfit(x_axis, values, 1)[0]
        mean_abs_step = np.abs(np.diff(values)).mean()
    else:
        slope = 0.0
        mean_abs_step = 0.0

    return {
        "sample_count": int(values.size),
        "sample_mean": float(mean),
        "sample_std": float(std),
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
        "sample_skew": float(skew),
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
    df["sample_mean_minus_event"] = df["sample_mean"] - df["event_level_mean"]
    df["sample_std_ratio"] = safe_divide(df["sample_std"], df["event_stdv"])
    df["sample_range_ratio"] = safe_divide(df["sample_range"], df["event_stdv"])
    df["sample_first_last_diff"] = df["sample_last"] - df["sample_first"]
    df["sample_median_minus_model"] = df["sample_median"] - df["model_mean"]
    df["sample_q90_minus_q10"] = df["sample_q90"] - df["sample_q10"]
    df["kmer_match"] = (df["reference_kmer"] == df["model_kmer"]).astype(int)
    df["kmer_hamming"] = [
        hamming_distance(a, b)
        for a, b in zip(df["reference_kmer"], df["model_kmer"])
    ]
    df["reference_center_base"] = df["reference_kmer"].astype(str).str[2]
    df["model_center_base"] = df["model_kmer"].astype(str).str[2]
    df["reference_gc_count"] = df["reference_kmer"].astype(str).apply(
        lambda text: sum(base in {"G", "C"} for base in text)
    )
    df["model_gc_count"] = df["model_kmer"].astype(str).apply(
        lambda text: sum(base in {"G", "C"} for base in text)
    )
    df["gc_count_diff"] = df["reference_gc_count"] - df["model_gc_count"]

    for index in range(5):
        ref_col = f"reference_base_{index}"
        model_col = f"model_base_{index}"
        match_col = f"base_match_{index}"
        df[ref_col] = df["reference_kmer"].astype(str).str[index]
        df[model_col] = df["model_kmer"].astype(str).str[index]
        df[match_col] = (df[ref_col] == df[model_col]).astype(int)

    df["is_mito"] = df["contig"].astype(str).eq("chrM").astype(int)
    return df


def get_feature_lists():
    numeric_cols = [
        "position", "read_index", "event_index", "event_level_mean", "event_stdv",
        "event_length", "model_mean", "model_stdv", "standardized_level",
        "start_idx", "end_idx", "sample_count", "sample_mean", "sample_std",
        "sample_min", "sample_max", "sample_median", "sample_q10", "sample_q25",
        "sample_q75", "sample_q90", "sample_iqr", "sample_range", "sample_first",
        "sample_last", "sample_slope", "sample_mean_abs_step", "sample_skew",
        "signal_diff", "abs_diff", "signal_ratio", "z_signal_diff", "event_span",
        "length_per_sample", "sample_mean_minus_model", "sample_mean_minus_event",
        "sample_std_ratio", "sample_range_ratio", "sample_first_last_diff",
        "sample_median_minus_model", "sample_q90_minus_q10", "kmer_match",
        "kmer_hamming", "reference_gc_count", "model_gc_count", "gc_count_diff",
        "base_match_0", "base_match_1", "base_match_2", "base_match_3",
        "base_match_4", "is_mito",
    ]

    categorical_cols = [
        "contig", "strand", "reference_kmer", "model_kmer",
        "reference_center_base", "model_center_base",
        "reference_base_0", "reference_base_1", "reference_base_2",
        "reference_base_3", "reference_base_4",
        "model_base_0", "model_base_1", "model_base_2", "model_base_3",
        "model_base_4",
    ]
    return numeric_cols, categorical_cols


def prepare_event_table(df):
    df = add_event_features(df)
    numeric_cols, categorical_cols = get_feature_lists()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df, numeric_cols, categorical_cols


def build_site_table(event_df, numeric_cols, categorical_cols):
    numeric_df = event_df[numeric_cols].copy()
    numeric_df["site"] = event_df["site"].values
    grouped = numeric_df.groupby("site")

    summary_tables = [
        grouped.mean().add_suffix("_mean"),
        grouped.std().fillna(0.0).add_suffix("_std"),
        grouped.min().add_suffix("_min"),
        grouped.max().add_suffix("_max"),
        grouped.median().add_suffix("_median"),
        grouped.quantile(0.10).add_suffix("_q10"),
        grouped.quantile(0.25).add_suffix("_q25"),
        grouped.quantile(0.75).add_suffix("_q75"),
        grouped.quantile(0.90).add_suffix("_q90"),
    ]

    site_df = pd.concat(summary_tables, axis=1)
    site_df["site_read_count"] = grouped.size().astype(float)
    site_df["site_event_index_span"] = (
        grouped["event_index"].max() - grouped["event_index"].min()
    ).astype(float)
    site_df["site_signal_diff_range"] = (
        grouped["signal_diff"].max() - grouped["signal_diff"].min()
    ).astype(float)
    site_df["site_abs_diff_mean"] = grouped["abs_diff"].mean().astype(float)
    site_df["site_sample_slope_mean"] = grouped["sample_slope"].mean().astype(float)
    site_df["site_kmer_match_fraction"] = grouped["kmer_match"].mean().astype(float)
    site_df["site_signal_ratio_std"] = grouped["signal_ratio"].std().fillna(0.0)
    site_df["site_z_signal_diff_max"] = grouped["z_signal_diff"].max().astype(float)
    site_df["site_z_signal_diff_min"] = grouped["z_signal_diff"].min().astype(float)
    site_df["site_length_per_sample_mean"] = (
        grouped["length_per_sample"].mean().astype(float)
    )

    site_labels = event_df.groupby("site").agg(
        label=("label", "max"),
        contig=("contig", "first"),
        strand=("strand", lambda s: s.mode().iat[0]),
        reference_kmer=("reference_kmer", lambda s: s.mode().iat[0]),
        model_kmer=("model_kmer", lambda s: s.mode().iat[0]),
        reference_center_base=("reference_center_base", lambda s: s.mode().iat[0]),
        model_center_base=("model_center_base", lambda s: s.mode().iat[0]),
        reference_base_0=("reference_base_0", lambda s: s.mode().iat[0]),
        reference_base_1=("reference_base_1", lambda s: s.mode().iat[0]),
        reference_base_2=("reference_base_2", lambda s: s.mode().iat[0]),
        reference_base_3=("reference_base_3", lambda s: s.mode().iat[0]),
        reference_base_4=("reference_base_4", lambda s: s.mode().iat[0]),
        model_base_0=("model_base_0", lambda s: s.mode().iat[0]),
        model_base_1=("model_base_1", lambda s: s.mode().iat[0]),
        model_base_2=("model_base_2", lambda s: s.mode().iat[0]),
        model_base_3=("model_base_3", lambda s: s.mode().iat[0]),
        model_base_4=("model_base_4", lambda s: s.mode().iat[0]),
    )

    site_df = site_df.join(site_labels, how="inner").reset_index()

    for col in categorical_cols:
        site_df[col] = site_df[col].astype("category")

    feature_cols = [col for col in site_df.columns if col not in {"site", "label"}]
    return site_df, feature_cols


def choose_model_name(choice):
    if choice == "auto":
        return "xgboost" if XGBClassifier is not None else "histgb"
    return choice


def get_param_grid(model_name):
    if model_name == "xgboost":
        return [
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 2,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 450,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_weight": 3,
                "reg_lambda": 4.0,
            },
            {
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.06,
                "subsample": 0.90,
                "colsample_bytree": 0.75,
                "min_child_weight": 1,
                "reg_lambda": 3.0,
            },
            {
                "n_estimators": 700,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 4,
                "reg_lambda": 6.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.90,
                "colsample_bytree": 0.70,
                "min_child_weight": 2,
                "reg_lambda": 5.0,
            },
        ]

    return [
        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.5,
        },
        {
            "learning_rate": 0.03,
            "max_depth": 8,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 25,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.07,
            "max_depth": 5,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "l2_regularization": 0.2,
        },
        {
            "learning_rate": 0.04,
            "max_depth": 7,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 10,
            "l2_regularization": 1.5,
        },
    ]


def build_model(model_name, params, random_state):
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            enable_categorical=True,
            random_state=random_state,
            n_jobs=-1,
            **params,
        )

    return HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=random_state,
        categorical_features="from_dtype",
        **params,
    )


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


def select_best_model(train_site_df, feature_cols, model_name):
    x_train = train_site_df[feature_cols]
    y_train = train_site_df["label"].to_numpy()

    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_result = None
    ranked_results = []

    for index, params in enumerate(get_param_grid(model_name), start=1):
        fold_prob = np.zeros(len(train_site_df), dtype=float)

        for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_train), start=1):
            model = build_model(model_name, params, RANDOM_STATE + fold_id)
            model.fit(x_train.iloc[fit_idx], train_site_df.iloc[fit_idx]["label"])
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

    ranked_results.sort(
        key=lambda row: (row["cv_auprc"], row["cv_f1"], row["cv_auroc"]),
        reverse=True,
    )
    best_result["top_results"] = [dict(row) for row in ranked_results[:3]]
    return best_result


def validate_site_labels(event_df):
    mixed_sites = int((event_df.groupby("site")["label"].nunique() > 1).sum())
    if mixed_sites:
        raise ValueError(
            f"Found {mixed_sites} sites with mixed labels. Each site must have one label."
        )


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
    event_df, numeric_cols, categorical_cols = prepare_event_table(raw_df)
    validate_site_labels(event_df)

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
    train_site_df, feature_cols = build_site_table(
        train_event_df, numeric_cols, categorical_cols
    )
    test_site_df, _ = build_site_table(test_event_df, numeric_cols, categorical_cols)

    model_name = choose_model_name(MODEL_CHOICE)
    print(f"\nTraining model: {model_name}")
    print(f"Train event rows: {len(train_event_df)}")
    print(f"Test event rows: {len(test_event_df)}")
    print(f"Train sites: {len(train_site_df)}")
    print(f"Test sites: {len(test_site_df)}")

    print("\nSelecting hyperparameters and threshold with site-level CV...")
    best_result = select_best_model(train_site_df, feature_cols, model_name)

    print("\nBest grouped-CV selection:")
    print(best_result)

    final_models = []
    for model_id, result in enumerate(best_result["top_results"], start=1):
        model = build_model(model_name, result["params"], RANDOM_STATE + model_id)
        model.fit(train_site_df[feature_cols], train_site_df["label"])
        final_models.append(model)

    site_prob = ensemble_predict(final_models, test_site_df[feature_cols])
    site_true = test_site_df["label"].to_numpy()
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
        "numeric_event_columns": numeric_cols,
        "categorical_event_columns": categorical_cols,
        "threshold": best_result["threshold"],
    }

    model_path = Path(f"{OUTPUT_PREFIX}.joblib")
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    print(f"\nSaved model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
