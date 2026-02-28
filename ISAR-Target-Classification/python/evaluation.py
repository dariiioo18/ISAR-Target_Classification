"""
Evaluation metrics and visualisation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """Return a dict with accuracy, precision, recall, and f1 (weighted)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_report(name, y_true, y_pred, class_names):
    """Print a full classification report for one model."""
    m = compute_metrics(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1-Score : {m['f1']:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return m


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_metrics_comparison(results: dict, save_path=None):
    """Bar chart comparing Accuracy / Precision / Recall / F1 across models.

    Parameters
    ----------
    results : dict[str, dict]  –  {model_name: metrics_dict}
    """
    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    friendly = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(model_names))
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (key, label) in enumerate(zip(metric_names, friendly)):
        vals = [results[m][key] for m in model_names]
        bars = ax.bar(x + i * bar_w, vals, width=bar_w, label=label)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + bar_w * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Evaluation Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot CNN accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
