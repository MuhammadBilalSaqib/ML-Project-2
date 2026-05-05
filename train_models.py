import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_ROOT = os.path.join("dataset", "chest_xray")
IMG_SIZE     = (64, 64)          # resize all images to 64×64 grayscale
CLASSES      = ["NORMAL", "PNEUMONIA"]
LABEL_MAP    = {c: i for i, c in enumerate(CLASSES)}   # NORMAL=0, PNEUMONIA=1


# ──────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────
def load_split(split: str, status_cb=None) -> tuple:
    """
    Load all images from  dataset/chest_xray/<split>/NORMAL  and
                           dataset/chest_xray/<split>/PNEUMONIA
    Returns (X, y) as numpy arrays.
    """
    X, y = [], []
    split_dir = os.path.join(DATASET_ROOT, split)

    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpeg", ".jpg", ".png"))]

        for fname in files:
            path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(path).convert("L")   # grayscale
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                X.append(arr)
                y.append(LABEL_MAP[cls])
            except Exception:
                pass   # skip corrupt files

        if status_cb:
            status_cb(f"  Loaded {len(files)} {cls} images from {split}/")

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# Confusion-matrix plot
# ──────────────────────────────────────────────
def save_confusion_matrix(cm, title: str, filepath: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "f1_score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
    }


def confusion_matrix_details(cm) -> dict:
    """Extract TP / TN / FP / FN from a 2×2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return {
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def train_and_evaluate(status_cb=None) -> dict:
    os.makedirs("static/confusion_matrices", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Load data ────────────────────────────
    if status_cb:
        status_cb("Loading training images…")
    X_train, y_train = load_split("train", status_cb)

    if status_cb:
        status_cb("Loading test images…")
    X_test, y_test = load_split("test", status_cb)

    if status_cb:
        status_cb(
            f"Dataset loaded — {len(X_train)} train, {len(X_test)} test samples. "
            f"Features per image: {X_train.shape[1]} ({IMG_SIZE[0]}×{IMG_SIZE[1]} px)"
        )

    results = {}

    # ── 1. KNN ────────────────────────────────
    if status_cb:
        status_cb("Training K-Nearest Neighbours (KNN, k=5)…")

    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    if status_cb:
        status_cb("KNN trained. Running predictions…")
    knn_pred = knn.predict(X_test)
    knn_time = round(time.time() - t0, 2)

    knn_cm = confusion_matrix(y_test, knn_pred)
    save_confusion_matrix(
        knn_cm,
        "KNN — Confusion Matrix\n(Chest X-Ray: NORMAL vs PNEUMONIA)",
        "static/confusion_matrices/knn_cm.png",
    )
    joblib.dump(knn, "models/knn_model.pkl")

    results["knn"] = {
        "label": "K-Nearest Neighbours",
        "short": "KNN",
        **compute_metrics(y_test, knn_pred),
        **confusion_matrix_details(knn_cm),
        "training_time": knn_time,
        "confusion_matrix": knn_cm.tolist(),
        "image": "confusion_matrices/knn_cm.png",
        "report": classification_report(y_test, knn_pred,
                                        target_names=CLASSES, output_dict=True),
    }
    if status_cb:
        status_cb(f"KNN done — Accuracy: {results['knn']['accuracy']}%  |  Time: {knn_time}s")

    # ── 2. Decision Tree ──────────────────────
    if status_cb:
        status_cb("Training Decision Tree (max_depth=20)…")

    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_time = round(time.time() - t0, 2)

    dt_cm = confusion_matrix(y_test, dt_pred)
    save_confusion_matrix(
        dt_cm,
        "Decision Tree — Confusion Matrix\n(Chest X-Ray: NORMAL vs PNEUMONIA)",
        "static/confusion_matrices/dt_cm.png",
    )
    joblib.dump(dt, "models/dt_model.pkl")

    results["decision_tree"] = {
        "label": "Decision Tree",
        "short": "DT",
        **compute_metrics(y_test, dt_pred),
        **confusion_matrix_details(dt_cm),
        "training_time": dt_time,
        "confusion_matrix": dt_cm.tolist(),
        "image": "confusion_matrices/dt_cm.png",
        "report": classification_report(y_test, dt_pred,
                                        target_names=CLASSES, output_dict=True),
    }
    if status_cb:
        status_cb(f"Decision Tree done — Accuracy: {results['decision_tree']['accuracy']}%  |  Time: {dt_time}s")

    # ── 3. Naive Bayes ────────────────────────
    if status_cb:
        status_cb("Training Gaussian Naive Bayes…")

    t0 = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_time = round(time.time() - t0, 2)

    nb_cm = confusion_matrix(y_test, nb_pred)
    save_confusion_matrix(
        nb_cm,
        "Naive Bayes — Confusion Matrix\n(Chest X-Ray: NORMAL vs PNEUMONIA)",
        "static/confusion_matrices/nb_cm.png",
    )
    joblib.dump(nb, "models/nb_model.pkl")

    results["naive_bayes"] = {
        "label": "Naive Bayes",
        "short": "NB",
        **compute_metrics(y_test, nb_pred),
        **confusion_matrix_details(nb_cm),
        "training_time": nb_time,
        "confusion_matrix": nb_cm.tolist(),
        "image": "confusion_matrices/nb_cm.png",
        "report": classification_report(y_test, nb_pred,
                                        target_names=CLASSES, output_dict=True),
    }
    if status_cb:
        status_cb(f"Naive Bayes done — Accuracy: {results['naive_bayes']['accuracy']}%  |  Time: {nb_time}s")

    # ── Best model ────────────────────────────
    best_key = max(
        ["knn", "decision_tree", "naive_bayes"],
        key=lambda k: results[k]["accuracy"],
    )
    results["best_model"] = best_key
    results["dataset_info"] = {
        "name": "Chest X-Ray (Pneumonia)",
        "classes": CLASSES,
        "train_normal":    int(np.sum(y_train == 0)),
        "train_pneumonia": int(np.sum(y_train == 1)),
        "test_normal":     int(np.sum(y_test  == 0)),
        "test_pneumonia":  int(np.sum(y_test  == 1)),
        "total_train":     int(len(X_train)),
        "total_test":      int(len(X_test)),
        "image_size":      f"{IMG_SIZE[0]}×{IMG_SIZE[1]} (grayscale)",
        "features":        int(X_train.shape[1]),
    }

    with open("static/results.json", "w") as f:
        json.dump(results, f, indent=2)

    if status_cb:
        status_cb(
            f"All done!  Best model → {results[best_key]['label']} "
            f"({results[best_key]['accuracy']}% accuracy)"
        )

    return results


if __name__ == "__main__":
    train_and_evaluate(status_cb=print)
