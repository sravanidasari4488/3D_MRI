"""
2D dataset loader for slice-based training on BraTS-style HDF5 data.

Responsibilities:
- Read volume IDs from split files (e.g., MRI/splits/train_ids.txt)
- For each volume ID, load all corresponding slice .h5 files:
      image: (240, 240, 4)
      mask : (240, 240, 3)
- Normalize images to float32 range [0, 1]
- Convert masks to float32
- Apply slice-level balancing per volume:
      * keep all slices with tumor (mask sum > 0)
      * randomly sample empty slices to match the number of tumor slices
- Concatenate balanced slices across all volumes in the split.

This module does NOT implement any training logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import h5py
import numpy as np


ArrayPair = Tuple[np.ndarray, np.ndarray]


def _parse_volume_number(volume_id: str) -> int:
    """
    Extract integer volume number from an ID such as:
        "volume_001.h5" -> 1
        "volume_65.h5"  -> 65
        "volume_65"     -> 65
    """
    vol = volume_id.strip()
    m = re.match(r"volume_(\d+)(?:\.h5)?$", vol, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse volume number from ID: {volume_id!r}")
    return int(m.group(1))


def _find_slice_files_for_volume(data_dir: Path, volume_number: int) -> List[Path]:
    """
    Find all slice .h5 files for a given volume number.

    Looks for files matching:
        volume_<N>_slice_*.h5
    in `data_dir`. If none are found, also checks data_dir / "data".
    """
    data_root = Path(data_dir)

    pattern = f"volume_{volume_number}_slice_*.h5"
    slice_files = sorted(
        data_root.glob(pattern),
        key=lambda p: int(re.search(r"slice_(\d+)", p.name).group(1)),
    )

    if not slice_files:
        data_subdir = data_root / "data"
        if data_subdir.exists():
            slice_files = sorted(
                data_subdir.glob(pattern),
                key=lambda p: int(re.search(r"slice_(\d+)", p.name).group(1)),
            )
            data_root = data_subdir

    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for volume {volume_number}.\n"
            f"Expected pattern: {pattern}\n"
            f"Searched in: {data_root}"
        )

    return slice_files


def _load_slices_for_volume(data_dir: Path, volume_number: int) -> ArrayPair:
    """
    Load all slices for one volume as 2D samples.

    Returns
    -------
    X_vol : np.ndarray, shape (Ns, 240, 240, 4)
    Y_vol : np.ndarray, shape (Ns, 240, 240, 3)
    """
    slice_files = _find_slice_files_for_volume(data_dir, volume_number)

    n_slices = len(slice_files)
    if n_slices == 0:
        raise RuntimeError(f"No slice files found for volume {volume_number}")

    # Use float16 internally to reduce memory footprint; convert to float32 on return.
    X_vol = np.empty((n_slices, 240, 240, 4), dtype=np.float16)
    Y_vol = np.empty((n_slices, 240, 240, 3), dtype=np.float16)

    for idx, sf in enumerate(slice_files):
        with h5py.File(sf, "r") as f:
            img = f["image"][...]  # (240, 240, 4)
            msk = f["mask"][...]   # (240, 240, 3)

        if img.shape != (240, 240, 4):
            raise ValueError(
                f"Unexpected image shape {img.shape} in {sf}; expected (240, 240, 4)"
            )
        if msk.shape != (240, 240, 3):
            raise ValueError(
                f"Unexpected mask shape {msk.shape} in {sf}; expected (240, 240, 3)"
            )

        X_vol[idx] = img  # cast to float16
        Y_vol[idx] = msk  # cast to float16

    # Normalize images to [0, 1] per volume.
    min_val = float(X_vol.min())
    max_val = float(X_vol.max())
    if max_val > min_val:
        X_vol = (X_vol - min_val) / (max_val - min_val)
    else:
        X_vol = np.zeros_like(X_vol, dtype=np.float16)

    return X_vol.astype(np.float32), Y_vol.astype(np.float32)


def _balance_slices_for_volume(
    X_vol: np.ndarray,
    Y_vol: np.ndarray,
    rng: np.random.Generator,
) -> ArrayPair:
    """
    Apply slice-level balancing for a single volume.

    - Keep all slices with tumor (mask sum > 0)
    - Randomly sample empty slices (sum == 0) to match tumor slice count
    """
    if X_vol.shape[0] != Y_vol.shape[0]:
        raise ValueError(
            f"Mismatched slice counts: X={X_vol.shape[0]}, Y={Y_vol.shape[0]}"
        )

    # Compute per-slice tumor presence from the full 3-channel mask.
    mask_sums = Y_vol.sum(axis=(1, 2, 3))
    tumor_idx = np.where(mask_sums > 0)[0]
    background_idx = np.where(mask_sums == 0)[0]

    n_tumor = int(tumor_idx.size)
    n_background = int(background_idx.size)

    if n_tumor == 0 or n_background == 0:
        # Degenerate case: no tumor or no background; keep all slices.
        keep_idx = np.arange(X_vol.shape[0])
    else:
        if n_background <= n_tumor:
            bg_keep = background_idx
        else:
            bg_keep = rng.choice(background_idx, size=n_tumor, replace=False)

        keep_idx = np.concatenate([tumor_idx, bg_keep])
        rng.shuffle(keep_idx)

    X_bal = X_vol[keep_idx]
    Y_bal = Y_vol[keep_idx]

    return X_bal, Y_bal


def _read_volume_ids(split_file: Path) -> List[str]:
    """Read non-empty volume IDs from a split file."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    lines = split_file.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_split_2d(
    split_file: str | Path,
    data_dir: str | Path,
    random_state: int = 42,
) -> ArrayPair:
    """
    Load and balance 2D slices for a given split.

    Parameters
    ----------
    split_file : str or Path
        Path to a text file containing one volume ID per line
        (e.g., MRI/splits/train_ids.txt).
    data_dir : str or Path
        Root directory containing BraTS slice .h5 files. If a "data"
        subdirectory is present, it will be used automatically.
    random_state : int
        Seed for reproducible background-slice sampling.

    Returns
    -------
    X : np.ndarray
        All balanced image slices, shape (N, 240, 240, 4).
    Y : np.ndarray
        All balanced mask slices, shape (N, 240, 240, 3).
    """
    split_path = Path(split_file)
    data_root = Path(data_dir)

    volume_ids = _read_volume_ids(split_path)
    if not volume_ids:
        raise ValueError(f"No volume IDs found in split file: {split_path}")

    rng = np.random.default_rng(random_state)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    total_slices = 0

    print(f"Loading 2D slices for split: {split_path}")
    print(f"Data directory: {data_root}")
    print(f"Number of volumes in split: {len(volume_ids)}")

    for vol_id in volume_ids:
        vol_num = _parse_volume_number(vol_id)
        print(f"\nVolume {vol_id} (number {vol_num}):")

        X_vol, Y_vol = _load_slices_for_volume(data_root, vol_num)
        print(f"  Loaded slices: {X_vol.shape[0]}")

        # Balance slices for this volume.
        X_bal, Y_bal = _balance_slices_for_volume(X_vol, Y_vol, rng)
        print(
            f"  Balanced slices: {X_bal.shape[0]} "
            f"(images: {X_bal.shape}, masks: {Y_bal.shape})"
        )

        X_list.append(X_bal)
        Y_list.append(Y_bal)
        total_slices += X_bal.shape[0]

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    print(f"\nTotal balanced slices loaded: {total_slices}")
    print(f"Final X shape: {X.shape}")
    print(f"Final Y shape: {Y.shape}")

    return X, Y


def load_train_val_test_2d(
    data_dir: str | Path,
    splits_dir: str | Path = "splits",
    random_state: int = 42,
) -> Tuple[ArrayPair, ArrayPair, ArrayPair]:
    """
    Convenience helper to load train / val / test splits in one call.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing BraTS slice .h5 files.
    splits_dir : str or Path
        Directory where train_ids.txt, val_ids.txt, and test_ids.txt live.
    random_state : int
        Seed for reproducible sampling (used consistently across splits).

    Returns
    -------
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    splits_root = Path(splits_dir)

    rng = np.random.default_rng(random_state)

    def _load_with_rng(name: str) -> ArrayPair:
        # Use a fresh seed derived from the base random_state for each split.
        sub_seed = int(rng.integers(0, 2**31 - 1))
        split_path = splits_root / f"{name}_ids.txt"
        return load_split_2d(split_path, data_dir, random_state=sub_seed)

    train = _load_with_rng("train")
    val = _load_with_rng("val")
    test = _load_with_rng("test")

    return train, val, test


if __name__ == "__main__":
    # Minimal CLI for ad-hoc testing.
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and balance 2D BraTS slices for a given split."
    )
    parser.add_argument(
        "split_file",
        type=str,
        help="Path to split file (e.g., MRI/splits/train_ids.txt)",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root directory containing volume_*_slice_*.h5 files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for background-slice sampling (default: 42).",
    )

    args = parser.parse_args()

    X, Y = load_split_2d(args.split_file, args.data_dir, random_state=args.seed)
    # Shapes and counts are printed inside load_split_2d

