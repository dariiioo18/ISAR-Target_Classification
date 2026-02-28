"""
Data loading and preprocessing utilities for ISAR images.
"""

import os
import numpy as np
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import IMG_SIZE, COLOUR_MODE, TEST_SIZE, RANDOM_STATE


def load_images_from_folder(folder: str):
    """Load PNG images from *folder*, returning flattened pixel arrays and labels.

    Labels are extracted from the filename prefix (everything before the first
    underscore, e.g. ``Caja`` from ``Caja12_gauss_0.10.png``).

    Returns
    -------
    X : np.ndarray of shape (N, H*W)
    y_labels : np.ndarray of str
    """
    images, labels = [], []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")

    filenames = sorted(os.listdir(folder))
    for fname in filenames:
        if not fname.lower().endswith(".png"):
            continue
        try:
            label = fname.split("_")[0]
            # Remove trailing digits from label (e.g. "Caja12" -> "Caja")
            label = "".join(c for c in label if not c.isdigit())

            img = Image.open(os.path.join(folder, fname)).convert(COLOUR_MODE)
            img = img.resize(IMG_SIZE)
            images.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")

    if not images:
        raise RuntimeError(f"No valid PNG images found in {folder}")

    return np.array(images, dtype=np.float32), np.array(labels)


def prepare_data(folder: str):
    """Full data preparation pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test  – NumPy arrays (numeric labels, normalised)
        label_encoder                      – fitted LabelEncoder
        class_names                        – list[str]
    """
    X, y_labels = load_images_from_folder(folder)

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Normalise pixel values to [0, 1]
    X = X / 255.0

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    print(f"Dataset loaded: {len(X)} images, {len(le.classes_)} classes")
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Distribution: {dict(Counter(y_labels))}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": le,
        "class_names": list(le.classes_),
    }
