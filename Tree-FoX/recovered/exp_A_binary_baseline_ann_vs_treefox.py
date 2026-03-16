# exp_A_binary_baseline_ann_vs_treefox_ON_YOUR_DATA_GROUPRULE.py
# Binary experiment:
# - NoKnowledge: simple ANN on full feature vector
# - WithKnowledge: Tree-FoX grouped modeling using knowledge-guided grouping
#
# Grouping uses scalar vs vector rule and bucket/group logic.

import os
import json
import time
import re
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model


SEED = 2
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "../model/dataset/merge_csv_samples_20240809_filtered_categories.csv"
CAT_COL = "categories"
KNOWLEDGE_JSON = "../model_v2/knowledge_20240809_v2_model_20240809_200313.json"

OUT_DIR = "exp_A_outputs_binary_ann_vs_treefox"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUT_DIR, "results_A_binary_baselineANN_vs_treefox.json")
CM_ANN_PNG = os.path.join(OUT_DIR, "cm_binary_baseline_ann.png")
CM_TREEFOX_PNG = os.path.join(OUT_DIR, "cm_binary_treefox.png")


def to_binary_label(cat: str) -> int:
    cat = str(cat).strip().lower()
    return 0 if cat == "benign" else 1


def clean_feature_columns(df: pd.DataFrame, target_col: str):
    cols = [c for c in df.columns if c != target_col]
    # Drop obvious non-feature text columns if present
    drop_candidates = ["label", "Label", "File Name", "File Name_x", "filename", "file_name"]
    cols = [c for c in cols if c not in drop_candidates]
    return cols


def coerce_numeric_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X


def save_confusion_matrix(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def build_simple_ann(input_dim: int):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_simple_ann(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = build_simple_ann(X_train_sc.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train_sc, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        verbose=0,
        callbacks=callbacks
    )

    y_prob = model.predict(X_test_sc, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }

    return metrics, y_pred, y_prob


def load_knowledge_groups(feature_cols, knowledge_json_path):
    """
    Best-effort reconstruction.
    If JSON exists and contains feature-group information, use it.
    Otherwise fallback to rule-based grouping.
    """
    if os.path.exists(knowledge_json_path):
        try:
            with open(knowledge_json_path, "r", encoding="utf-8") as f:
                knowledge = json.load(f)

            groups = defaultdict(list)

            # Generic recovery logic for several likely JSON shapes
            if isinstance(knowledge, dict):
                if "groups" in knowledge and isinstance(knowledge["groups"], dict):
                    for g, feats in knowledge["groups"].items():
                        for feat in feats:
                            if feat in feature_cols:
                                groups[g].append(feat)
                else:
                    for feat in feature_cols:
                        matched = False
                        for g, vals in knowledge.items():
                            if isinstance(vals, list) and feat in vals:
                                groups[g].append(feat)
                                matched = True
                        if not matched:
                            pass

            groups = {k: v for k, v in groups.items() if len(v) > 0}
            if groups:
                return groups
        except Exception as e:
            print(f"Warning: could not parse knowledge JSON. Falling back. Error: {e}")

    return build_groups_by_rule(feature_cols)


def build_groups_by_rule(feature_cols):
    """
    Scalar vs vector / bucket-style rule reconstruction.
    """
    groups = defaultdict(list)

    for feat in feature_cols:
        f = feat.lower()

        if "imports" in f:
            groups["imports"].append(feat)
        elif "exports" in f:
            groups["exports"].append(feat)
        elif "strings" in f:
            groups["strings"].append(feat)
        elif "." in feat:
            prefix = feat.split(".", 1)[0].strip()
            if prefix:
                groups[prefix].append(feat)
            else:
                groups["other"].append(feat)
        else:
            groups["scalar"].append(feat)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


def train_group_models(X_train, y_train, X_test, groups):
    group_preds = {}
    group_probs = {}

    for group_name, feats in groups.items():
        if len(feats) == 0:
            continue

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[feats])
        Xte = scaler.transform(X_test[feats])

        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(Xtr, y_train)

        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        group_probs[group_name] = prob
        group_preds[group_name] = pred

    return group_preds, group_probs


def aggregate_treefox_predictions(group_probs):
    """
    Mean probability aggregation as a safe reconstruction.
    """
    group_names = list(group_probs.keys())
    probs = np.column_stack([group_probs[g] for g in group_names])
    final_prob = probs.mean(axis=1)
    final_pred = (final_prob >= 0.5).astype(int)
    return final_pred, final_prob


def evaluate_binary(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }


def main():
    t0 = time.time()

    df = pd.read_csv(CSV_PATH)
    if CAT_COL not in df.columns:
        raise ValueError(f"Missing target column: {CAT_COL}")

    df[CAT_COL] = df[CAT_COL].astype(str).str.strip()
    df["binary_label"] = df[CAT_COL].apply(to_binary_label)

    feature_cols = clean_feature_columns(df, "binary_label")
    X = coerce_numeric_features(df, feature_cols)
    y = df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # NoKnowledge: simple ANN
    ann_metrics, ann_pred, ann_prob = train_simple_ann(X_train, y_train, X_test, y_test)
    save_confusion_matrix(y_test, ann_pred, CM_ANN_PNG, "Binary Baseline ANN")

    # WithKnowledge: Tree-FoX grouped modeling
    groups = load_knowledge_groups(list(X.columns), KNOWLEDGE_JSON)
    group_preds, group_probs = train_group_models(X_train, y_train, X_test, groups)
    treefox_pred, treefox_prob = aggregate_treefox_predictions(group_probs)
    treefox_metrics = evaluate_binary(y_test, treefox_pred, treefox_prob)
    save_confusion_matrix(y_test, treefox_pred, CM_TREEFOX_PNG, "Binary Tree-FoX")

    results = {
        "config": {
            "seed": SEED,
            "csv_path": CSV_PATH,
            "knowledge_json": KNOWLEDGE_JSON,
            "n_features": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_groups": int(len(groups)),
            "group_names": list(groups.keys())
        },
        "baseline_ann": ann_metrics,
        "treefox_with_knowledge": treefox_metrics,
        "runtime_sec": float(time.time() - t0)
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()