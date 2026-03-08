"""
Preprocessing pipeline for 3D multi-modal brain MRI (BraTS-style HDF5 data).

Goals:
- Load T1, T2, and FLAIR modalities for a given volume
- Normalize each modality volume independently
- Center-crop each modality to (128, 128, 128)
- Stack modalities along the channel axis → output shape (128, 128, 128, 3)
- Provide a `combine_modalities()` helper
- Visualize one axial slice across the three channels

This script focuses ONLY on clean preprocessing; no training or patch extraction.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_volume_h5(data_dir, volume_number, modality_channel):
    """
    Load a full 3D MRI volume (single modality) from BraTS-style HDF5 slices.

    Each HDF5 file contains:
        'image': (H, W, 4) with channels [T1, T2, FLAIR, T1CE]

    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing `volume_XXX_slice_YYY.h5` files
    volume_number : int
        Volume ID to load (e.g., 1, 2, 41)
    modality_channel : int
        Channel index for the modality:
            0 → T1
            1 → T2
            2 → FLAIR

    Returns
    -------
    volume : np.ndarray, shape (H, W, D), dtype float32
        Loaded 3D volume for the requested modality.
    """
    data_path = Path(data_dir)

    # If there is a nested "data" directory, use it (matches Kaggle layout)
    if (data_path / "data").exists():
        data_path = data_path / "data"

    pattern = f"volume_{volume_number}_slice_*.h5"
    slice_files = sorted(
        data_path.glob(pattern),
        key=lambda p: int(p.name.split("_slice_")[-1].split(".")[0]),
    )

    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for volume {volume_number} in {data_path}\n"
            f"Expected pattern: {pattern}"
        )

    # Inspect first slice to get spatial size
    with h5py.File(slice_files[0], "r") as f:
        h, w, _ = f["image"].shape

    depth = len(slice_files)
    volume = np.zeros((h, w, depth), dtype=np.float32)

    for z, slice_file in enumerate(slice_files):
        with h5py.File(slice_file, "r") as f:
            img = f["image"][:]  # (H, W, 4)
            volume[:, :, z] = img[:, :, modality_channel]

    return volume


def normalize_volume(volume, eps: float = 1e-6):
    """
    Normalize a 3D volume independently using z-score on non-zero voxels.

    Steps:
    - Compute mean and std over voxels where volume != 0
    - If not enough non-zero voxels, fall back to all voxels
    - Return normalized volume as float32
    """
    vol = volume.astype(np.float32)

    mask = vol != 0
    if np.any(mask):
        vals = vol[mask]
    else:
        vals = vol.reshape(-1)

    mean = float(vals.mean())
    std = float(vals.std())

    if std < eps:
        std = 1.0  # avoid division by ~0; volume is nearly constant

    vol_norm = (vol - mean) / std
    return vol_norm.astype(np.float32)


def crop_center(volume, target_shape=(128, 128, 128)):
    """
    Center-crop (or pad) a 3D volume to the target shape.

    If the input is smaller than the target along any axis, zero-padding
    is applied symmetrically to reach the target size.

    Parameters
    ----------
    volume : np.ndarray, shape (H, W, D)
        Input 3D volume.
    target_shape : tuple of int
        Desired (H, W, D) after cropping/padding.

    Returns
    -------
    cropped : np.ndarray, shape target_shape
        Cropped (and possibly padded) volume.
    """
    assert volume.ndim == 3, "Expected 3D volume"
    th, tw, td = target_shape
    h, w, d = volume.shape

    # Compute start/end indices for cropping
    def compute_bounds(size, target):
        if size >= target:
            start = (size - target) // 2
            end = start + target
            return start, end
        else:
            # Will pad later
            return 0, size

    sh, eh = compute_bounds(h, th)
    sw, ew = compute_bounds(w, tw)
    sd, ed = compute_bounds(d, td)

    cropped = volume[sh:eh, sw:ew, sd:ed]

    # Pad if needed
    pad_h = max(th - cropped.shape[0], 0)
    pad_w = max(tw - cropped.shape[1], 0)
    pad_d = max(td - cropped.shape[2], 0)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        pad_before = (pad_h // 2, pad_w // 2, pad_d // 2)
        pad_after = (
            pad_h - pad_before[0],
            pad_w - pad_before[1],
            pad_d - pad_before[2],
        )
        cropped = np.pad(
            cropped,
            pad_width=(
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=0,
        )

    assert cropped.shape == target_shape, f"Got {cropped.shape}, expected {target_shape}"
    return cropped.astype(np.float32)


def combine_modalities(t1_vol, t2_vol, flair_vol):
    """
    Stack three preprocessed modalities along a channel axis.

    Expected input shapes: (128, 128, 128) for each modality.
    Output shape: (128, 128, 128, 3) with channels-last ordering.

    Channel order:
        0 → T1
        1 → T2
        2 → FLAIR
    """
    assert (
        t1_vol.shape == t2_vol.shape == flair_vol.shape
    ), "All modalities must have the same shape before stacking"

    stacked = np.stack([t1_vol, t2_vol, flair_vol], axis=-1)
    # stacked.shape == (H, W, D, 3)
    return stacked.astype(np.float32)


def visualize_stacked_slice(stacked_volume, slice_index=None):
    """
    Visualize one axial slice from the stacked multi-modal volume.

    Parameters
    ----------
    stacked_volume : np.ndarray, shape (H, W, D, 3)
        Stacked volume with channels [T1, T2, FLAIR].
    slice_index : int, optional
        Axial slice index along the depth (D). If None, use the center slice.
    """
    assert stacked_volume.ndim == 4 and stacked_volume.shape[-1] == 3
    h, w, d, _ = stacked_volume.shape

    if slice_index is None:
        slice_index = d // 2

    t1_slice = stacked_volume[:, :, slice_index, 0]
    t2_slice = stacked_volume[:, :, slice_index, 1]
    flair_slice = stacked_volume[:, :, slice_index, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(t1_slice, cmap="gray", origin="lower")
    axes[0].set_title(f"T1 - Slice {slice_index}", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(t2_slice, cmap="gray", origin="lower")
    axes[1].set_title(f"T2 - Slice {slice_index}", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    im3 = axes[2].imshow(flair_slice, cmap="gray", origin="lower")
    axes[2].set_title(f"FLAIR - Slice {slice_index}", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.suptitle("Multi-modal MRI (T1, T2, FLAIR) - Single Axial Slice", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    """
    Example CLI for preprocessing a single BraTS volume.

    Steps:
    - Load T1, T2, FLAIR volumes
    - Normalize each independently
    - Center-crop to (128, 128, 128)
    - Stack into (128, 128, 128, 3)
    - Visualize one slice
    """
    data_directory = input("Enter path to HDF5 data directory: ").strip()
    data_directory = data_directory.strip('"').strip("'")

    volume_number = int(input("Enter volume number (e.g., 1, 2, 41): ").strip())

    print("\nLoading modalities...")
    t1 = load_volume_h5(data_directory, volume_number, modality_channel=0)
    t2 = load_volume_h5(data_directory, volume_number, modality_channel=1)
    flair = load_volume_h5(data_directory, volume_number, modality_channel=2)

    print(f"T1 shape:    {t1.shape}, range=({t1.min():.2f}, {t1.max():.2f})")
    print(f"T2 shape:    {t2.shape}, range=({t2.min():.2f}, {t2.max():.2f})")
    print(f"FLAIR shape: {flair.shape}, range=({flair.min():.2f}, {flair.max():.2f})")

    print("\nNormalizing volumes (per modality)...")
    t1_norm = normalize_volume(t1)
    t2_norm = normalize_volume(t2)
    flair_norm = normalize_volume(flair)

    print("Cropping volumes to (128, 128, 128) centered region...")
    target_shape = (128, 128, 128)
    t1_crop = crop_center(t1_norm, target_shape)
    t2_crop = crop_center(t2_norm, target_shape)
    flair_crop = crop_center(flair_norm, target_shape)

    stacked = combine_modalities(t1_crop, t2_crop, flair_crop)
    print(f"\nStacked volume shape: {stacked.shape} (H, W, D, C)")

    # Visualize one slice
    print("\nVisualizing one axial slice from stacked channels...")
    visualize_stacked_slice(stacked, slice_index=None)


if __name__ == "__main__":
    main()


