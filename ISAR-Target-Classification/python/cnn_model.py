"""
CNN definition, training, and prediction for ISAR target classification.
"""

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    IMG_SIZE,
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_EARLY_STOPPING_PATIENCE,
)


def build_cnn(num_classes: int):
    """Build a simple 3-block CNN for grayscale ISAR images."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _reshape_for_cnn(X):
    """Reshape flat vectors (N, H*W) -> (N, H, W, 1)."""
    return X.astype("float32").reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)


def train_cnn(X_train, y_train, X_test, y_test, num_classes: int):
    """Train the CNN with data augmentation and early stopping.

    Returns (model, history).
    """
    X_train_cnn = _reshape_for_cnn(X_train)
    X_test_cnn = _reshape_for_cnn(X_test)

    model = build_cnn(num_classes)
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(X_train_cnn)

    early_stop = EarlyStopping(
        patience=CNN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        datagen.flow(X_train_cnn, y_train, batch_size=CNN_BATCH_SIZE),
        validation_data=(X_test_cnn, y_test),
        epochs=CNN_EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    return model, history


def predict_cnn(model, X):
    """Return class predictions for flat input X."""
    X_cnn = _reshape_for_cnn(X)
    return np.argmax(model.predict(X_cnn), axis=1)
