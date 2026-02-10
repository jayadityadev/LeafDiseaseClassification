"""
Leaf condition classification (prediction only).
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# Simple in-memory cache for loaded Keras models
_MODEL_CACHE: dict[str, keras.Model] = {}

# Label mapping must match training order
LABEL_NAMES = [
    "Healthy_Leaf",
    "Healthy_Nut",
    "Mahali_Koleroga",
    "Yellow_Leaf",
    "Ring_Spot",
    "Bud_Rot",
]

IMG_SIZE = (224, 224)


def _load_model(model_path: str | Path) -> keras.Model:
    model_path = str(model_path)
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached
    model = keras.models.load_model(model_path)
    _MODEL_CACHE[model_path] = model
    return model


def predict_with_localization(
    image_path: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.70,
) -> dict:
    """
    Predict leaf condition class.
    """
    image_path = str(image_path)
    model = _load_model(model_path)

    # Load and preprocess image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    except cv2.error:
        img_resized = np.array(Image.fromarray(img_rgb).resize(IMG_SIZE, Image.Resampling.LANCZOS))

    img_array = img_resized.astype(np.float32)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    pred_probs = model.predict(img_batch, verbose=0)[0]
    pred_class = int(np.argmax(pred_probs))
    pred_label = LABEL_NAMES[pred_class]
    confidence = float(pred_probs[pred_class] * 100)

    epsilon = 1e-10
    entropy = float(-np.sum(pred_probs * np.log(pred_probs + epsilon)))
    max_entropy = float(-np.log(1.0 / len(pred_probs)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    is_uncertain = (confidence < (confidence_threshold * 100)) or (normalized_entropy > 0.4)

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "entropy": normalized_entropy,
        "confidence_threshold": confidence_threshold * 100,
        "probabilities": {LABEL_NAMES[i]: float(p * 100) for i, p in enumerate(pred_probs)},
        "warning_message": (
            f"Low confidence ({confidence:.1f}%) - possible out-of-scope condition or image quality issue. "
            "Manual review recommended."
        ) if is_uncertain else None,
    }


if __name__ == "__main__":
    example_image = Path("data/leaf_dataset/test/Healthy_Leaf").glob("*.jpg")
    sample = next(example_image, None)
    if sample is None:
        raise SystemExit("No sample images found.")

    model_candidate = Path("models/leaf/current/densenet121").glob("*.keras")
    model_path = next(model_candidate, None)
    if model_path is None:
        raise SystemExit("No model found under models/leaf/current/densenet121")

    results = predict_with_localization(sample, model_path)
    print(results)
