"""
train.py — Model Training Script
Trains a MobileNetV2-based binary classifier to detect
damaged vs not_damaged smartphones.

Usage:
    python train.py
"""

import os
import tensorflow as tf
import tensorflow as tf
MobileNetV2 = tf.keras.applications.MobileNetV2
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Model = tf.keras.Model
Adam = tf.keras.optimizers.Adam
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"
MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "model.h5")
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 1e-4
NUM_CLASSES   = 2


def build_data_generators():
    """
    Create ImageDataGenerators for training and validation.
    Applies augmentation on training data and rescaling on both splits.
    """
    # Training generator: augmentation + rescaling
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
    )

    # Validation generator: only rescaling (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model():
    """
    Build transfer-learning model using MobileNetV2 as frozen base.
    Adds a custom classification head for 2 output classes.
    """
    # Load pretrained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze all base layers — we only train our custom head
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train():
    """
    Main training function: builds data generators, constructs the model,
    trains it, and saves the result to disk.
    """
    # Ensure model output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n📦  Loading dataset from:", DATASET_DIR)
    train_gen, val_gen = build_data_generators()

    print(f"✅  Classes found: {train_gen.class_indices}")
    print(f"    Training samples  : {train_gen.samples}")
    print(f"    Validation samples: {val_gen.samples}\n")

    print("🏗️   Building model …")
    model = build_model()
    model.summary()

    print(f"\n🚀  Training for {EPOCHS} epochs …\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"\n💾  Model saved to: {MODEL_PATH}")

    # Final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    print(f"\n📊  Final Training Accuracy  : {final_train_acc:.4f}")
    print(f"    Final Validation Accuracy: {final_val_acc:.4f}")
    print("\n✅  Training complete!\n")


if __name__ == "__main__":
    train()
