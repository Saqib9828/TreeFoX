import os
import re
import json
import time
import joblib
import random
import numpy as np
import pandas as pd

from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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


SEED = 2
random.seed(SEED)
np.random.seed(SEED)

CSV_PATH = "../model/dataset/merge_csv_samples_20240809_filtered_categories.csv"
TARGET_COL = "categories"

RUN_NAME = time.strftime("tree_fox_run_%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("tree_fox_runs", RUN_NAME)
MODEL_DIR = os.path.join(OUT_DIR, "models")
ART_DIR = os.path.join(OUT_DIR, "artifacts")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


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


def build_groups(feature_cols):
    groups = defaultdict(list)
    known_prefixes = {
        "pslist", "dlllist", "handles", "ldrmodules", "malfind",
        "psxview", "modules", "svcscan", "callbacks"
    }

    for feat in feature_cols:
        fl = feat.lower()
        if "imports" in fl:
            groups["imports"].append(feat)
        elif "exports" in fl:
            groups["exports"].append(feat)
        elif "strings" in fl:
            groups["strings"].append(feat)
        elif "." in feat:
            prefix = feat.split(".", 1)[0].strip().lower()
            if prefix in known_prefixes:
                groups[prefix].append(feat)
            else:
                groups["other"].append(feat)
        else:
            groups["scalar"].append(feat)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


def train_group_models(X_train, y_train, X_val, groups):
    group_models = {}
    group_scalers = {}
    val_group_preds = {}
    val_group_probs = {}

    for group_name, feats in groups.items():
        Xtr = X_train[feats].copy()
        Xva = X_val[feats].copy()

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr)
        Xva_sc = scaler.transform(Xva)

        clf = LogisticRegression(
            max_iter=3000,
            multi_class="auto",
            random_state=SEED
        )
        clf.fit(Xtr_sc, y_train)

        pred = clf.predict(Xva_sc)
        prob = clf.predict_proba(Xva_sc)

        group_models[group_name] = clf
        group_scalers[group_name] = scaler
        val_group_preds[group_name] = pred
        val_group_probs[group_name] = prob

    return group_models, group_scalers, val_group_preds, val_group_probs


def weighted_mode_aggregate(group_preds, group_probs, class_names, top_groups=2):
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
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df["major_category"] = df[TARGET_COL].apply(map_major_category)
    df = df[df["major_category"].isin(VALID_MULTICLASS)].copy()

    feature_cols = clean_feature_columns(df, "major_category")
    X = coerce_numeric(df, feature_cols)
    y_raw = df["major_category"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    groups = build_groups(list(X.columns))

    # save metadata before training
    save_json({
        "seed": SEED,
        "csv_path": CSV_PATH,
        "target_col": TARGET_COL,
        "class_names": class_names,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "groups": {k: len(v) for k, v in groups.items()}
    }, os.path.join(ART_DIR, "run_config.json"))

    save_json(groups, os.path.join(ART_DIR, "groups.json"))
    pd.DataFrame({"feature": list(X.columns)}).to_csv(
        os.path.join(ART_DIR, "all_features.csv"), index=False
    )

    pd.DataFrame({
        "row_index": idx_train,
        "split": "train"
    }).to_csv(os.path.join(ART_DIR, "train_indices.csv"), index=False)

    pd.DataFrame({
        "row_index": idx_test,
        "split": "test"
    }).to_csv(os.path.join(ART_DIR, "test_indices.csv"), index=False)

    # train per-group models
    group_models, group_scalers, test_group_preds, test_group_probs = train_group_models(
        X_train, y_train, X_test, groups
    )

    # aggregate
    y_pred, y_conf = weighted_mode_aggregate(
        test_group_preds,
        test_group_probs,
        class_names=class_names,
        top_groups=2
    )

    # metrics
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

    save_json(metrics, os.path.join(ART_DIR, "metrics_test.json"))

    # predictions
    pred_df = pd.DataFrame({
        "row_index": idx_test,
        "y_true_id": y_test,
        "y_pred_id": y_pred,
        "y_true": le.inverse_transform(y_test),
        "y_pred": le.inverse_transform(y_pred),
        "confidence_score": y_conf
    })
    pred_df.to_csv(os.path.join(ART_DIR, "test_predictions.csv"), index=False)

    # plots
    plot_cm(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        out_path=os.path.join(PLOT_DIR, "confusion_matrix_test.png"),
        title="Tree-FoX Test Confusion Matrix"
    )

    # save label encoder
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # save models and scalers
    for g in groups.keys():
        joblib.dump(group_models[g], os.path.join(MODEL_DIR, f"{g}_model.pkl"))
        joblib.dump(group_scalers[g], os.path.join(MODEL_DIR, f"{g}_scaler.pkl"))

    print("\nTraining finished.")
    print("Saved run to:", OUT_DIR)
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"]
    }, indent=2))


if __name__ == "__main__":
    main()