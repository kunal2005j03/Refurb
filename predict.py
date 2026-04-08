"""
predict.py — Inference / Prediction Module
Loads the trained model and classifies a single image as
'damaged' or 'not_damaged', along with a confidence score.

Usage (standalone):
    python predict.py path/to/image.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_PATH = "model/model.h5"
IMG_SIZE   = (224, 224)

# Class index → label mapping
# Must match the alphabetical order ImageDataGenerator assigns:
#   0 → damaged   |   1 → not_damaged
CLASS_LABELS = {0: "damaged", 1: "not_damaged"}


def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load and return the saved Keras model from disk.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Loaded Keras model.
    """
    model = load_model(model_path)
    return model


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize, normalize, and expand the PIL image to model input shape.

    Args:
        img: A PIL Image object.

    Returns:
        NumPy array of shape (1, 224, 224, 3), values in [0, 1].
    """
    # Resize to the expected input size
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    # Convert to numpy array and normalize to [0, 1]
    img_array = keras_image.img_to_array(img) / 255.0

    # Add batch dimension → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(img: Image.Image, model=None) -> dict:
    """
    Run inference on a single PIL image.

    Args:
        img  : PIL Image to classify.
        model: Optional pre-loaded Keras model (loaded from disk if None).

    Returns:
        dict with keys:
            'label'      — 'damaged' or 'not_damaged'
            'confidence' — float in [0, 1]
            'all_scores' — raw softmax scores for both classes
    """
    if model is None:
        model = load_trained_model()

    # Preprocess
    img_array = preprocess_image(img)

    # Predict
    predictions = model.predict(img_array, verbose=0)   # shape: (1, 2)
    scores      = predictions[0]                         # shape: (2,)

    # Determine winning class
    class_idx  = int(np.argmax(scores))
    label      = CLASS_LABELS[class_idx]
    confidence = float(scores[class_idx])

    return {
        "label"     : label,
        "confidence": confidence,
        "all_scores": {CLASS_LABELS[i]: float(scores[i]) for i in range(len(scores))},
    }


# ─────────────────────────────────────────────
# CLI entry-point for quick testing
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    img      = Image.open(img_path)

    print(f"\n🔍  Running prediction on: {img_path}")
    result = predict(img)

    print(f"\n📋  Result:")
    print(f"    Label      : {result['label']}")
    print(f"    Confidence : {result['confidence']:.2%}")
    print(f"    All scores : {result['all_scores']}")

    status = "✅ Good Condition" if result["label"] == "not_damaged" else "⚠️  Damaged"
    print(f"\n    Final Status: {status}\n")
