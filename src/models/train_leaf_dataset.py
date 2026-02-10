"""
Train a DenseNet121 model for 6-class leaf condition classification.
Uses the existing CSV splits under data/leaf_dataset/*_split.csv.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_CSV = PROJECT_ROOT / "data/leaf_dataset/train_split.csv"
VAL_CSV = PROJECT_ROOT / "data/leaf_dataset/val_split.csv"
TEST_CSV = PROJECT_ROOT / "data/leaf_dataset/test_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/leaf/current/densenet121"
REPORTS_DIR = PROJECT_ROOT / "outputs/leaf/reports"

# Training configuration
IMG_SIZE = (224, 224)
RESIZE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_LR = 1e-4
NUM_CLASSES = 6
USE_CLASS_WEIGHTS = False
CLASS_NAMES = [
    "Healthy_Leaf",
    "Healthy_Nut",
    "Mahali_Koleroga",
    "Yellow_Leaf",
    "Ring_Spot",
    "Bud_Rot",
]

# GPU configuration (prefer GPU, allow memory growth)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _configure_gpu() -> None:
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU ENABLED: Found {len(physical_devices)} GPU(s)")
            for i, gpu in enumerate(physical_devices):
                print(f"   GPU {i}: {gpu.name}")
            if os.environ.get("ENABLE_MIXED_PRECISION", "1") == "1":
                try:
                    from tensorflow.keras import mixed_precision
                    mixed_precision.set_global_policy("mixed_float16")
                    print("✅ Mixed precision enabled (mixed_float16)")
                except Exception as exc:
                    print(f"⚠️  Mixed precision not enabled: {exc}")
        else:
            print("⚠️  No GPU detected - Training will use CPU")
    except Exception as exc:
        print(f"⚠️  GPU configuration error: {exc}")
        print("   Attempting to continue with CPU...")


def _normalize_path(path_str: str) -> Path:
    # CSV uses backslashes; normalize for current OS
    normalized = Path(path_str.replace("\\", "/"))
    if normalized.is_absolute():
        return normalized
    return (PROJECT_ROOT / normalized).resolve()


def _build_dataset(csv_path: Path, batch_size: int, training: bool) -> tf.data.Dataset:
    df = pd.read_csv(csv_path)
    filepaths = [_normalize_path(fp).as_posix() for fp in df["filepath"].tolist()]
    labels = df["label"].astype(int).tolist()

    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, RESIZE_SIZE, method="lanczos3")
        if training:
            img = tf.image.random_crop(img, size=(IMG_SIZE[0], IMG_SIZE[1], 3))
        else:
            img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0], IMG_SIZE[1])
        img = preprocess_input(img)
        return img, tf.one_hot(label, NUM_CLASSES)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augment = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )

        def _augment(img, label):
            return augment(img, training=True), label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape=(224, 224, 3)):
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model, base_model


def train_leaf_densenet121():
    _configure_gpu()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("=" * 80)
    print("🍃 LEAF CONDITION CLASSIFICATION (DenseNet121)")
    print("=" * 80)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    steps_per_epoch = max(1, math.ceil(len(train_df) / BATCH_SIZE))
    validation_steps = max(1, math.ceil(len(val_df) / BATCH_SIZE))

    train_ds = _build_dataset(TRAIN_CSV, BATCH_SIZE, training=True)
    val_ds = _build_dataset(VAL_CSV, BATCH_SIZE, training=False)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_counts = train_df["label"].value_counts().to_dict()
        total = len(train_df)
        class_weight = {int(cls): total / (len(class_counts) * count) for cls, count in class_counts.items()}

    model, base_model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = OUTPUT_DIR / f"leaf_densenet121_best_{timestamp}.keras"
    final_model_path = OUTPUT_DIR / f"leaf_densenet121_final_{timestamp}.keras"

    callbacks = [
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        CSVLogger(str(OUTPUT_DIR / f"training_log_{timestamp}.csv")),
    ]

    frozen_epochs = max(1, EPOCHS - 20)
    history = model.fit(
        train_ds,
        epochs=frozen_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Fine-tune: unfreeze upper layers while keeping BatchNorm frozen
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"],
    )

    fine_tune_epochs = EPOCHS - frozen_epochs
    model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(str(final_model_path))
    print(f"✅ Saved final model to: {final_model_path}")

    # Evaluate on test split
    print("\n🔍 Evaluating on test split...")
    test_ds = _build_dataset(TEST_CSV, BATCH_SIZE, training=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        flipped = tf.image.flip_left_right(batch_imgs)
        preds_flipped = model.predict(flipped, verbose=0)
        preds = (preds + preds_flipped) / 2.0
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_path = REPORTS_DIR / f"classification_report_{timestamp}.txt"
    report_path.write_text(report)
    print(f"✅ Classification report saved: {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Leaf Dataset Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = REPORTS_DIR / f"confusion_matrix_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    train_leaf_densenet121()
