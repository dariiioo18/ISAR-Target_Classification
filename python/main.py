#!/usr/bin/env python3
"""
ISAR Target Classification – Main Pipeline
===========================================

Usage
-----
    # Run with default data directory (../data)
    python main.py

    # Specify a custom image folder
    python main.py --data /path/to/isar/images

    # Run hyper-parameter grid search before training
    python main.py --grid-search

    # Skip CNN training (ML models only)
    python main.py --skip-cnn
"""

import argparse
import sys

from data_loader import prepare_data
from ml_models import train_all_models, run_grid_search
from cnn_model import train_cnn, predict_cnn
from evaluation import (
    print_report,
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_training_history,
)
from config import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="ISAR target classification pipeline",
    )
    parser.add_argument(
        "--data", type=str, default=DATA_DIR,
        help="Path to the folder containing ISAR PNG images",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run hyper-parameter grid search before training",
    )
    parser.add_argument(
        "--skip-cnn", action="store_true",
        help="Skip CNN training (useful on machines without GPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading and preprocessing images ...")
    data = prepare_data(args.data)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    class_names = data["class_names"]
    n_classes = len(class_names)

    # ── 2. Optional grid search ─────────────────────────────────────────────
    if args.grid_search:
        print("\n[2/4] Running hyper-parameter grid search ...")
        run_grid_search(X_train, y_train)
    else:
        print("\n[2/4] Skipping grid search (use --grid-search to enable)")

    # ── 3. Train ML models ──────────────────────────────────────────────────
    print("\n[3/4] Training ML models ...")
    models = train_all_models(X_train, y_train, n_classes)

    results = {}
    for name, entry in models.items():
        y_pred = entry["predict"](X_test)
        metrics = print_report(name, y_test, y_pred, class_names)
        results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, class_names, title=f"{name} – Confusion Matrix")

    # ── 4. Train CNN ────────────────────────────────────────────────────────
    if not args.skip_cnn:
        print("\n[4/4] Training CNN ...")
        cnn, history = train_cnn(X_train, y_train, X_test, y_test, n_classes)

        y_pred_cnn = predict_cnn(cnn, X_test)
        cnn_metrics = print_report("CNN", y_test, y_pred_cnn, class_names)
        results["CNN"] = cnn_metrics

        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred_cnn, class_names, title="CNN – Confusion Matrix")
    else:
        print("\n[4/4] CNN training skipped (--skip-cnn)")

    # ── 5. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:20s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
    print("=" * 60)

    plot_metrics_comparison(results)


if __name__ == "__main__":
    main()
