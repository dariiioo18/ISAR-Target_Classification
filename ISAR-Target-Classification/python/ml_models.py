"""
Traditional machine-learning classifiers for ISAR target recognition.

Includes SVM, Decision Tree, Random Forest, and an adapted K-Means baseline.
"""

import numpy as np
from collections import Counter

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

from config import (
    SVM_PARAMS, DT_PARAMS, RF_PARAMS,
    KMEANS_N_INIT, RANDOM_STATE,
    SVM_GRID, DT_GRID, RF_GRID,
)


# ── Hyper-parameter search ──────────────────────────────────────────────────

def run_grid_search(X_train, y_train):
    """Run grid search for SVM, Decision Tree, and Random Forest.

    Returns a dict mapping model name -> GridSearchCV result object.
    """
    searches = {}

    configs = [
        ("SVM", SVC(random_state=RANDOM_STATE), SVM_GRID),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE), DT_GRID),
        ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), RF_GRID),
    ]

    for name, estimator, grid in configs:
        print(f"  Grid search: {name} ...")
        gs = GridSearchCV(estimator, grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        searches[name] = gs
        print(f"    Best params : {gs.best_params_}")
        print(f"    Best CV acc : {gs.best_score_:.4f}")

    return searches


# ── Model training ──────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    model = SVC(**SVM_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(**DT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_train, y_train, n_classes):
    """Train K-Means and map clusters to the most frequent true label.

    Returns (model, cluster_to_label mapping).
    """
    model = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    model.fit(X_train)

    cluster_labels = model.predict(X_train)
    cluster_to_label = {}
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        most_common = Counter(y_train[mask]).most_common(1)[0][0]
        cluster_to_label[cid] = most_common

    return model, cluster_to_label


def predict_kmeans(model, cluster_to_label, X):
    """Map K-Means cluster predictions to supervised labels."""
    clusters = model.predict(X)
    return np.array([cluster_to_label.get(c, -1) for c in clusters])


# ── Convenience function ────────────────────────────────────────────────────

def train_all_models(X_train, y_train, n_classes):
    """Train every ML model and return a dict of {name: (model, y_pred_test_fn)}.

    Returns
    -------
    dict : name -> dict with 'model' and 'predict' callable(X) -> y_pred
    """
    models = {}

    svm = train_svm(X_train, y_train)
    models["SVM"] = {"model": svm, "predict": svm.predict}

    dt = train_decision_tree(X_train, y_train)
    models["Decision Tree"] = {"model": dt, "predict": dt.predict}

    rf = train_random_forest(X_train, y_train)
    models["Random Forest"] = {"model": rf, "predict": rf.predict}

    km, mapping = train_kmeans(X_train, y_train, n_classes)
    models["K-Means"] = {
        "model": km,
        "predict": lambda X, _m=km, _mp=mapping: predict_kmeans(_m, _mp, X),
    }

    return models
