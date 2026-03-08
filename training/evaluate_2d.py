"""
Evaluation script for 2D U-Net models on BraTS-style 2D slices.

This script:
- Loads a trained 2D U-Net (`best_model.h5`)
- For each test volume ID in `test_ids.txt`:
      * loads all 2D slices (images and masks)
      * predicts a 2D mask for each slice
      * stacks predictions into a 3D volume (slices form the depth dimension)
      * computes a multi-class Dice score per volume
- Prints Dice per volume, then mean and standard deviation across volumes.

Evaluation is performed per volume, not per slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from training.dataset_loader_2d import (
    _parse_volume_number,
    _load_slices_for_volume,
    _read_volume_ids,
)


def dice_per_volume_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute multi-class Dice for a single volume.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth one-hot labels, shape (D, H, W, C) or (N, H, W, C).
    y_pred : np.ndarray
        Predicted one-hot labels, same shape as y_true.
    eps : float
        Smoothing term to avoid division by zero.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # Sum over all spatial dimensions and batch/slice dim; keep classes separate.
    axes = tuple(range(y_true.ndim - 1))  # all dims except last (classes)
    intersection = np.sum(y_true * y_pred, axis=axes)
    denominator = np.sum(y_true, axis=axes) + np.sum(y_pred, axis=axes)

    dice_per_class = (2.0 * intersection + eps) / (denominator + eps)
    return float(np.mean(dice_per_class))


def main() -> None:
    """
    Evaluate a trained 2D U-Net on test volumes and report per-volume Dice.
    """
    # --- Paths and configuration ---
    splits_dir_input = input(
        "Enter path to splits directory (containing test_ids.txt) [default: splits]: "
    ).strip()
    if not splits_dir_input:
        splits_dir_input = "splits"
    splits_dir = Path(splits_dir_input.strip('"').strip("'"))

    test_split = splits_dir / "test_ids.txt"

    data_dir_input = input(
        "Enter path to HDF5 data directory "
        "(where volume_XXX_slice_YYY.h5 lives): "
    ).strip()
    data_dir = Path(data_dir_input.strip('"').strip("'"))

    model_path_input = input(
        "Enter path to trained model (best_model.h5) [default: best_model.h5]: "
    ).strip()
    if not model_path_input:
        model_path_input = "best_model.h5"
    model_path = Path(model_path_input.strip('"').strip("'"))

    if not splits_dir.is_dir():
        raise SystemExit(f"Splits directory not found: {splits_dir}")
    if not test_split.exists():
        raise SystemExit(f"Test split file not found: {test_split}")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    # --- Load model (no need to compile for inference) ---
    print(f"\nLoading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # --- Read test volume IDs ---
    volume_ids = _read_volume_ids(test_split)
    if not volume_ids:
        raise SystemExit(f"No volume IDs found in test split: {test_split}")

    print(f"\nNumber of test volumes: {len(volume_ids)}")

    dice_scores: List[float] = []

    for vol_id in volume_ids:
        vol_num = _parse_volume_number(vol_id)
        print(f"\nEvaluating volume {vol_id} (number {vol_num})...")

        # Load all slices for this volume (no balancing).
        X_slices, Y_slices = _load_slices_for_volume(data_dir, vol_num)
        # X_slices: (Ns, 240, 240, 4), Y_slices: (Ns, 240, 240, 3)

        # Run model prediction.
        pred_probs = model.predict(X_slices, batch_size=8, verbose=0)
        # pred_probs: (Ns, 240, 240, C)

        num_classes = pred_probs.shape[-1]

        # Convert ground truth to labels by argmax over channels, then to one-hot.
        gt_bin = (Y_slices > 0.5).astype(np.float32)
        gt_labels = np.argmax(gt_bin, axis=-1)  # (Ns, H, W)
        gt_one_hot = tf.one_hot(gt_labels, depth=num_classes).numpy().astype(np.float32)

        # Convert predictions to labels (argmax of softmax) and to one-hot.
        pred_labels = np.argmax(pred_probs, axis=-1)
        pred_one_hot = tf.one_hot(pred_labels, depth=num_classes).numpy().astype(
            np.float32
        )

        # Treat slices dimension as depth; compute multi-class Dice per volume.
        dice = dice_per_volume_multiclass(gt_one_hot, pred_one_hot)
        dice_scores.append(dice)

        print(f"  Dice (multi-class) for {vol_id}: {dice:.4f}")

    # --- Aggregate metrics across volumes ---
    dice_array = np.asarray(dice_scores, dtype=np.float32)
    mean_dice = float(dice_array.mean()) if dice_array.size > 0 else float("nan")
    std_dice = float(dice_array.std(ddof=0)) if dice_array.size > 0 else float("nan")

    print("\n=== 2D U-Net Evaluation (per-volume Dice) ===")
    print(f"Number of test volumes: {len(dice_scores)}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Std Dice:  {std_dice:.4f}")


if __name__ == "__main__":
    main()

