"""
End-to-end training script for 2D U-Net on BraTS-style 2D slices.

This script:
- Loads balanced 2D training and validation data using `dataset_loader_2d`
- Builds a 2D U-Net model from `unet2d_model`
- Trains the model with:
      learning rate = 1e-4
      batch size    = 8
      epochs        = 30 (with EarlyStopping)
- Tracks and prints training / validation Dice per epoch
- Saves the best-performing model as `best_model.h5`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from training.dataset_loader_2d import load_split_2d
from training.unet2d_model import build_unet_2d, dice_coefficient_multiclass


def main() -> None:
    """
    Train a 2D U-Net on BraTS 2D slices.
    """
    # --- I/O paths ---
    splits_dir_input = input(
        "Enter path to splits directory "
        "(containing train_ids.txt / val_ids.txt) [default: splits]: "
    ).strip()
    if not splits_dir_input:
        splits_dir_input = "splits"
    splits_dir = Path(splits_dir_input.strip('"').strip("'"))

    train_split = splits_dir / "train_ids.txt"
    val_split = splits_dir / "val_ids.txt"

    data_dir_input = input(
        "Enter path to HDF5 data directory "
        "(where volume_XXX_slice_YYY.h5 lives): "
    ).strip()
    data_dir = Path(data_dir_input.strip('"').strip("'"))

    if not splits_dir.is_dir():
        raise SystemExit(f"Splits directory not found: {splits_dir}")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # --- Load training and validation data ---
    print("\nLoading TRAIN data (balanced slices)...")
    X_train, Y_train = load_split_2d(train_split, data_dir, random_state=42)

    print("\nLoading VAL data (balanced slices)...")
    X_val, Y_val = load_split_2d(val_split, data_dir, random_state=123)

    # Ensure data is float32
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    Y_val = np.asarray(Y_val, dtype=np.float32)

    print("\nFinal dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Y_val:   {Y_val.shape}")

    # --- Build model ---
    input_shape = X_train.shape[1:]  # (240, 240, 4)
    print(f"\nBuilding 2D U-Net with input shape: {input_shape}")
    model = build_unet_2d(
        input_shape=input_shape,
        base_filters=32,
        num_classes=Y_train.shape[-1],
        learning_rate=1e-4,
    )

    print("\nModel summary:")
    model.summary(line_length=120)

    # --- Training configuration ---
    batch_size = 8
    epochs = 30

    # EarlyStopping on validation Dice (multi-class).
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coefficient_multiclass",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint_path = Path("best_model.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_dice_coefficient_multiclass",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # Callback to print Dice each epoch in a concise way.
    class DiceLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_dice = logs.get("dice_coefficient_multiclass")
            val_dice = logs.get("val_dice_coefficient_multiclass")
            if train_dice is not None and val_dice is not None:
                print(
                    f"Epoch {epoch + 1}: "
                    f"train Dice = {train_dice:.4f} | "
                    f"val Dice = {val_dice:.4f}"
                )

    print(
        f"\nStarting training for up to {epochs} epochs "
        f"(batch_size={batch_size})..."
    )
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, DiceLogger()],
    )

    print(f"\nBest model saved to: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()

