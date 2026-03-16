# exp_A_treefox_multiclass_mode.py
# Exp A: Tree-FoX multiclass (major category) using grouped features + MODE aggregation
# - NoKnowledge: group models + plain mode
# - WithKnowledge: feature scaling before grouping + knowledge-weighted mode
#
# Target: major category derived from Category column: Benign/Spyware/Ransomware/Trojan
# Split: 80/20 with seed=2
# Grouping: prefix before '.' in feature name
# (pslist, dlllist, handles, ldrmodules, malfind, psxview, modules, svcscan, callbacks, other)

import os
import json
import time
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score
)

import matplotlib.pyplot as plt
import seaborn as sns


SEED = 2
K_SLICES = 5
TOP_GROUPS = 2

CSV_PATH = "../model/dataset/merge_csv_samples_20240809_filtered_categories.csv"
CAT_COL = "categories"

OUT_DIR = "exp_A_outputs_multiclass_treefox_mode"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUT_DIR, "results_multiclass_treefox_mode.json")
CM_NOKNOWLEDGE_PNG = os.path.join(OUT_DIR, "cm_multiclass_noknowledge_mode.png")
CM_WITHKNOWLEDGE_PNG = os.path.join(OUT_DIR, "cm_multiclass_withknowledge_mode.png")


VALID_CLASSES = ["Benign", "Spyware", "Ransomware", "Trojan"]


def map_major_category(value: str) -> str:
    v = str(value).strip().lower()

    if v == "benign":
        return "Benign"
    if "spy" in v:
        return "Spyware"
    if "ransom" in v:
        return "Ransomware"
    if "trojan" in v:
        return "Trojan"

    # fallback, adapt if your categories had more variants
    return v.capitalize()


def clean_feature_columns(df: pd.DataFrame, target_col: str):
    cols = [c for c in df.columns if c != target_col]
    drop_candidates = ["label", "Label", "File Name", "File Name_x", "filename", "file_name"]
    cols = [c for c in cols if c not in drop_candidates]
    return cols


def coerce_numeric_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X


def build_prefix_groups(feature_cols):
    groups = defaultdict(list)

    allowed_prefixes = {
        "pslist", "dlllist", "handles", "ldrmodules", "malfind",
        "psxview", "modules", "svcscan", "callbacks"
    }

    for feat in feature_cols:
        if "." in feat:
            prefix = feat.split(".", 1)[0].strip().lower()
            if prefix in allowed_prefixes:
                groups[prefix].append(feat)
            else:
                groups["other"].append(feat)
        else:
            groups["other"].append(feat)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


def train_group_classifiers(X_train, y_train, X_test, groups, scale_before=True):
    group_preds = {}
    group_confs = {}

    for group_name, feats in groups.items():
        if len(feats) == 0:
            continue

        Xtr = X_train[feats].copy()
        Xte = X_test[feats].copy()

        if scale_before:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        clf = LogisticRegression(
            max_iter=3000,
            multi_class="auto",
            random_state=SEED
        )
        clf.fit(Xtr, y_train)

        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)

        group_preds[group_name] = pred
        group_confs[group_name] = prob

    return group_preds, group_confs


def aggregate_plain_mode(group_preds):
    group_names = list(group_preds.keys())
    n_samples = len(next(iter(group_preds.values())))
    final_pred = []

    for i in range(n_samples):
        votes = [group_preds[g][i] for g in group_names]
        vote_count = Counter(votes)
        final_pred.append(vote_count.most_common(1)[0][0])

    return np.array(final_pred)


def aggregate_weighted_mode(group_preds, group_confs, top_groups=TOP_GROUPS):
    group_names = list(group_preds.keys())
    n_samples = len(next(iter(group_preds.values())))
    final_pred = []

    for i in range(n_samples):
        sample_scores = []

        for g in group_names:
            pred_label = group_preds[g][i]
            pred_conf = float(np.max(group_confs[g][i]))
            sample_scores.append((g, pred_label, pred_conf))

        sample_scores = sorted(sample_scores, key=lambda x: x[2], reverse=True)[:top_groups]

        weighted_votes = defaultdict(float)
        for _, label, conf in sample_scores:
            weighted_votes[label] += conf

        best_label = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)[0][0]
        final_pred.append(best_label)

    return np.array(final_pred)


def save_confusion_matrix(y_true, y_pred, class_names, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def evaluate_multiclass(y_true, y_pred, class_names):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred,
            labels=class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    }


def main():
    t0 = time.time()

    df = pd.read_csv(CSV_PATH)
    if CAT_COL not in df.columns:
        raise ValueError(f"Missing target column: {CAT_COL}")

    df[CAT_COL] = df[CAT_COL].astype(str).str.strip()
    df["major_category"] = df[CAT_COL].apply(map_major_category)

    df = df[df["major_category"].isin(VALID_CLASSES)].copy()

    feature_cols = clean_feature_columns(df, "major_category")
    X = coerce_numeric_features(df, feature_cols)
    y = df["major_category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    groups = build_prefix_groups(list(X.columns))

    # NoKnowledge: group models + plain mode
    nk_group_preds, nk_group_confs = train_group_classifiers(
        X_train, y_train, X_test, groups, scale_before=False
    )
    y_pred_nk = aggregate_plain_mode(nk_group_preds)
    nk_metrics = evaluate_multiclass(y_test, y_pred_nk, VALID_CLASSES)
    save_confusion_matrix(
        y_test, y_pred_nk, VALID_CLASSES,
        CM_NOKNOWLEDGE_PNG,
        "Multiclass Tree-FoX NoKnowledge, Plain Mode"
    )

    # WithKnowledge: feature scaling before grouping + knowledge-weighted mode
    wk_group_preds, wk_group_confs = train_group_classifiers(
        X_train, y_train, X_test, groups, scale_before=True
    )
    y_pred_wk = aggregate_weighted_mode(wk_group_preds, wk_group_confs, top_groups=TOP_GROUPS)
    wk_metrics = evaluate_multiclass(y_test, y_pred_wk, VALID_CLASSES)
    save_confusion_matrix(
        y_test, y_pred_wk, VALID_CLASSES,
        CM_WITHKNOWLEDGE_PNG,
        "Multiclass Tree-FoX WithKnowledge, Weighted Mode"
    )

    results = {
        "config": {
            "seed": SEED,
            "csv_path": CSV_PATH,
            "target_classes": VALID_CLASSES,
            "n_features": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "group_names": list(groups.keys()),
            "top_groups": TOP_GROUPS
        },
        "no_knowledge_plain_mode": nk_metrics,
        "with_knowledge_weighted_mode": wk_metrics,
        "runtime_sec": float(time.time() - t0)
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()