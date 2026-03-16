import os
import json
import joblib
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# set this to your saved run
RUN_DIR = r"tree_fox_runs/tree_fox_run_YYYYMMDD_HHMMSS"

MODEL_DIR = os.path.join(RUN_DIR, "models")
ART_DIR = os.path.join(RUN_DIR, "artifacts")
PLOT_DIR = os.path.join(RUN_DIR, "plots")
TEST_OUT_DIR = os.path.join(RUN_DIR, "test_outputs")

os.makedirs(TEST_OUT_DIR, exist_ok=True)

CSV_PATH = "../model/dataset/merge_csv_samples_20240809_filtered_categories.csv"
TARGET_COL = "categories"

VALID_MULTICLASS = ["Benign", "Spyware", "Ransomware", "Trojan"]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_cm(y_true, y_pred, labels, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def map_major_category(value):
    v = str(value).strip().lower()
    if v == "benign":
        return "Benign"
    if "spy" in v:
        return "Spyware"
    if "ransom" in v:
        return "Ransomware"
    if "trojan" in v:
        return "Trojan"
    return v.capitalize()


def clean_feature_columns(df, target_col):
    ignore_cols = {
        target_col, "label", "Label",
        "File Name", "File Name_x", "filename", "file_name"
    }
    return [c for c in df.columns if c not in ignore_cols]


def coerce_numeric(df, feature_cols):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X


def weighted_mode_aggregate(group_preds, group_probs, top_groups=2):
    group_names = list(group_preds.keys())
    n = len(next(iter(group_preds.values())))
    final_pred = []
    final_conf = []

    for i in range(n):
        sample_votes = []
        for g in group_names:
            pred_lbl = group_preds[g][i]
            prob_vec = group_probs[g][i]
            conf = float(np.max(prob_vec))
            sample_votes.append((g, pred_lbl, conf))

        sample_votes = sorted(sample_votes, key=lambda x: x[2], reverse=True)[:top_groups]

        agg = defaultdict(float)
        for _, lbl, conf in sample_votes:
            agg[lbl] += conf

        best = sorted(agg.items(), key=lambda x: x[1], reverse=True)[0]
        final_pred.append(best[0])
        final_conf.append(float(best[1]))

    return np.array(final_pred), np.array(final_conf)


def main():
    # load metadata
    with open(os.path.join(ART_DIR, "groups.json"), "r", encoding="utf-8") as f:
        groups = json.load(f)

    test_indices_df = pd.read_csv(os.path.join(ART_DIR, "test_indices.csv"))
    test_indices = test_indices_df["row_index"].tolist()

    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    class_names = list(le.classes_)

    # load data
    df = pd.read_csv(CSV_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df["major_category"] = df[TARGET_COL].apply(map_major_category)
    df = df[df["major_category"].isin(VALID_MULTICLASS)].copy().reset_index(drop=True)

    feature_cols = clean_feature_columns(df, "major_category")
    X = coerce_numeric(df, feature_cols)
    y_raw = df["major_category"].values
    y = le.transform(y_raw)

    # use saved test split
    X_test = X.iloc[test_indices].copy()
    y_test = y[test_indices]

    group_preds = {}
    group_probs = {}

    for g, feats in groups.items():
        model = joblib.load(os.path.join(MODEL_DIR, f"{g}_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, f"{g}_scaler.pkl"))

        Xg = X_test[feats].copy()
        Xg_sc = scaler.transform(Xg)

        group_preds[g] = model.predict(Xg_sc)
        group_probs[g] = model.predict_proba(Xg_sc)

    y_pred, y_conf = weighted_mode_aggregate(group_preds, group_probs, top_groups=2)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    }

    save_json(metrics, os.path.join(TEST_OUT_DIR, "metrics_retest.json"))

    pred_df = pd.DataFrame({
        "row_index": test_indices,
        "y_true_id": y_test,
        "y_pred_id": y_pred,
        "y_true": le.inverse_transform(y_test),
        "y_pred": le.inverse_transform(y_pred),
        "confidence_score": y_conf
    })
    pred_df.to_csv(os.path.join(TEST_OUT_DIR, "predictions_retest.csv"), index=False)

    plot_cm(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        out_path=os.path.join(TEST_OUT_DIR, "confusion_matrix_retest.png"),
        title="Tree-FoX Retest Confusion Matrix"
    )

    print("\nRetest finished.")
    print("Saved outputs to:", TEST_OUT_DIR)
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"]
    }, indent=2))


if __name__ == "__main__":
    main()