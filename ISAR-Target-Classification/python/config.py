"""
Global configuration for the ISAR classification pipeline.

Adjust these values to match your dataset location and experimental setup.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get(
    "ISAR_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data"),
)

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)
COLOUR_MODE = "L"  # Grayscale

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 1

# ---------------------------------------------------------------------------
# SVM hyper-parameters (from grid search)
# ---------------------------------------------------------------------------
SVM_PARAMS = {
    "C": 1,
    "gamma": 0.1,
    "kernel": "rbf",
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Decision Tree hyper-parameters (from grid search)
# ---------------------------------------------------------------------------
DT_PARAMS = {
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Random Forest hyper-parameters (from grid search)
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "criterion": "gini",
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
KMEANS_N_INIT = 10

# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_EARLY_STOPPING_PATIENCE = 5

# ---------------------------------------------------------------------------
# Grid-search parameter grids (used by --grid-search flag)
# ---------------------------------------------------------------------------
SVM_GRID = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": [0.001, 0.01, 0.1],
}

DT_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

RF_GRID = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
