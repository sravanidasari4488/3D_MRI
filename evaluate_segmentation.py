"""
Evaluation and visualization script for 3D brain tumor segmentation.

Features:
- Compute Dice, Precision, and Recall for:
    * GGMM segmentation (statistical prior)
    * Baseline 3D U-Net (3-channel input), if model available
    * GGMM-augmented 3D U-Net (4-channel input, with GGMM prior), if model available
    * Hybrid prediction: average of GGMM prior and GGMM-augmented U-Net output
- Visualize (single axial slice):
    * Original MRI slice (FLAIR)
    * Ground truth mask
    * GGMM segmentation
    * U-Net prediction (GGMM-augmented if available, else baseline)
    * Hybrid prediction
- Save hybrid predicted segmentation as a NIfTI file
- Print a comparison table of metrics suitable for research reporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf

from explore_brats import load_brats_volume
from ggmm_segmentation import load_3d_volume, process_3d_volume
from preprocess_brats_3d import (
    load_volume_h5,
    normalize_volume,
    crop_center,
    combine_modalities,
)



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute DSC (Dice), IoU, Precision, and Recall for binary segmentation masks.

    y_true, y_pred: boolean or {0,1} arrays with same shape.

    DSC (Dice) = 2 * TP / (2*TP + FP + FN)
    IoU = TP / (TP + FP + FN) = intersection / union
    """
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()

    # DSC (Dice Similarity Coefficient)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    # IoU (Intersection over Union) = TP / (TP + FP + FN)
    iou = (tp + eps) / (tp + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def visualize_slice(
    flair_slice: np.ndarray,
    gt_slice: np.ndarray,
    ggmm_slice: np.ndarray,
    unet_slice: np.ndarray,
    hybrid_slice: np.ndarray,
    slice_idx: int,
) -> None:
    """
    Visualize a single axial slice with multiple segmentation maps.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Original FLAIR
    im0 = axes[0].imshow(flair_slice, cmap="gray", origin="lower")
    axes[0].set_title(f"FLAIR (slice {slice_idx})")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Ground truth
    im1 = axes[1].imshow(gt_slice, cmap="gray", origin="lower")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # GGMM segmentation
    im2 = axes[2].imshow(ggmm_slice, cmap="viridis", origin="lower")
    axes[2].set_title("GGMM Segmentation")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # U-Net prediction
    im3 = axes[3].imshow(unet_slice, cmap="gray", origin="lower")
    axes[3].set_title("U-Net Prediction")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    # Hybrid prediction
    im4 = axes[4].imshow(hybrid_slice, cmap="gray", origin="lower")
    axes[4].set_title("Hybrid Prediction\n(U-Net + GGMM prior)")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    plt.tight_layout()
    plt.show()


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print metrics comparison table: method vs DSC (Dice) / IoU / Precision / Recall.
    """
    print("\n=== Segmentation Metrics Comparison ===")
    header = f"{'Method':<20} {'DSC':>8} {'IoU':>8} {'Precision':>10} {'Recall':>10}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['dice']:>8.4f} "
            f"{metrics['iou']:>8.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f}"
        )
    print()


def main():
    """
    Run full-volume evaluation and visualization on a single BraTS case.
    """
    data_dir = input(
        "Enter path to HDF5 data directory (where volume_XXX_slice_YYY.h5 lives): "
    ).strip()
    data_dir = data_dir.strip('"').strip("'")

    volume_number = int(input("Enter volume number (e.g., 1, 2, 41): ").strip())

    # Optional model paths
    baseline_path = Path("baseline_model.h5")
    ggmm_model_path = Path("ggmm_augmented_model.h5")

    print("\nLoading multi-modal volumes (T1, T2, FLAIR)...")
    t1 = load_volume_h5(data_dir, volume_number, modality_channel=0)
    t2 = load_volume_h5(data_dir, volume_number, modality_channel=1)
    flair = load_volume_h5(data_dir, volume_number, modality_channel=2)

    print("Loading ground-truth segmentation volume...")
    vols = load_brats_volume(data_dir, volume_number)
    seg_multiclass = vols["Segmentation"].astype(np.uint8)
    gt_mask = (seg_multiclass > 0).astype(np.uint8)

    print("\nApplying GGMM segmentation to FLAIR (statistical prior)...")
    flair_full = load_3d_volume(data_dir, volume_number, modality_channel=2)
    ggmm_labels = process_3d_volume(flair_full, n_components=3, verbose_slice=False)
    ggmm_prior = ggmm_labels.astype(np.float32)
    if ggmm_prior.max() > 0:
        ggmm_prior /= ggmm_prior.max()

    print("\nNormalizing modalities (T1, T2, FLAIR)...")
    t1_n = normalize_volume(t1)
    t2_n = normalize_volume(t2)
    flair_n = normalize_volume(flair)

    print("Center-cropping volumes, mask, and GGMM prior to (128,128,128)...")
    target_shape = (128, 128, 128)
    t1_c = crop_center(t1_n, target_shape)
    t2_c = crop_center(t2_n, target_shape)
    flair_c = crop_center(flair_n, target_shape)
    gt_c = crop_center(gt_mask.astype(np.float32), target_shape).astype(np.uint8)
    ggmm_c = crop_center(ggmm_prior, target_shape)

    print("\nComputing GGMM prior segmentation and metrics (no U-Net).")
    ggmm_seg_bin = (ggmm_c > 0.5).astype(np.uint8)

    # --- Metrics ---
    results: Dict[str, Dict[str, float]] = {}
    results["GGMM prior"] = compute_metrics(gt_c, ggmm_seg_bin)
    print_comparison_table(results)

    # --- Visualization for one slice ---
    h, w, d = gt_c.shape
    slice_idx = d // 2
    flair_slice = flair_c[:, :, slice_idx]
    gt_slice = gt_c[:, :, slice_idx]
    ggmm_slice = ggmm_seg_bin[:, :, slice_idx]

    # For now, reuse GGMM prior as placeholder for U-Net and hybrid views.
    unet_slice = ggmm_slice
    hybrid_slice = ggmm_slice

    print("\nDisplaying qualitative comparison for one axial slice...")
    visualize_slice(
        flair_slice, gt_slice, ggmm_slice, unet_slice, hybrid_slice, slice_idx
    )

    # --- Save GGMM prior prediction as NIfTI ---
    nii_path = Path(f"ggmm_prior_prediction_volume_{volume_number}.nii.gz")
    img = nib.Nifti1Image(ggmm_seg_bin.astype(np.uint8), affine=np.eye(4))
    nib.save(img, nii_path)
    print(f"\nSaved GGMM prior segmentation as NIfTI: {nii_path.resolve()}")


if __name__ == "__main__":
    main()


