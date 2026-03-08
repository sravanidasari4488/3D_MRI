"""
BraTS dataset split: volume-level train / validation / test.

Reads all volume identifiers from a directory, then randomly splits them into:
  - 70% train
  - 15% validation
  - 15% test

Uses a fixed random seed (42) for reproducibility.
Writes train_ids.txt, val_ids.txt, test_ids.txt (one volume ID per line).
No patching or training; this script only performs volume-level splitting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def get_volume_ids_from_directory(
    data_dir: str | Path,
    pattern: str = "volume_*_slice_*.h5",
) -> list[str]:
    """
    Discover unique volume IDs from a directory of BraTS-style files.

    Supports two naming conventions:
    1) One file per volume: volume_001.h5, volume_002.h5, ...
    2) Slices per volume: volume_1_slice_0.h5, volume_1_slice_1.h5, ...
       → unique IDs extracted and formatted as volume_001, volume_002, ...

    Returns a sorted list of volume identifiers (e.g. "volume_001.h5" or "volume_001").
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    # Try one-file-per-volume first: volume_XXX.h5
    single_files = sorted(data_dir.glob("volume_*.h5"))
    single_match = re.compile(r"volume_(\d+)\.h5$", re.IGNORECASE)

    volume_ids_set: set[str] = set()

    for f in single_files:
        m = single_match.match(f.name)
        if m:
            num = int(m.group(1))
            volume_ids_set.add(f"volume_{num:03d}.h5")

    if volume_ids_set:
        return sorted(volume_ids_set)

    # Otherwise assume slice-based: volume_X_slice_Y.h5
    slice_match = re.compile(r"volume_(\d+)_slice_\d+\.h5$", re.IGNORECASE)
    for f in data_dir.glob("volume_*_slice_*.h5"):
        m = slice_match.match(f.name)
        if m:
            num = int(m.group(1))
            volume_ids_set.add(f"volume_{num:03d}.h5")

    return sorted(volume_ids_set)


def split_volumes(
    volume_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split volume IDs into train / validation / test.

    Ratios should sum to 1.0. Uses a fixed random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    n = len(volume_ids)
    indices = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ids = [volume_ids[i] for i in train_idx]
    val_ids = [volume_ids[i] for i in val_idx]
    test_ids = [volume_ids[i] for i in test_idx]

    return train_ids, val_ids, test_ids


def save_splits(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    output_dir: str | Path,
) -> None:
    """Write one volume ID per line to train_ids.txt, val_ids.txt, test_ids.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in [
        ("train_ids.txt", train_ids),
        ("val_ids.txt", val_ids),
        ("test_ids.txt", test_ids),
    ]:
        path = output_dir / name
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        print(f"  Written {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split BraTS volume IDs into 70% train, 15% val, 15% test."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing volume files (e.g. volume_001.h5 or volume_*_slice_*.h5)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Directory for train_ids.txt, val_ids.txt, test_ids.txt (default: current)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
        help="Train fraction (default: 0.70)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.strip('"').strip("'")
    output_dir = args.output_dir.strip('"').strip("'")

    print(f"Scanning directory: {data_dir}")
    volume_ids = get_volume_ids_from_directory(data_dir)
    print(f"Found {len(volume_ids)} unique volume(s)")

    if len(volume_ids) == 0:
        raise SystemExit("No volume IDs found. Check data_dir and file naming.")

    train_ids, val_ids, test_ids = split_volumes(
        volume_ids,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_state=args.seed,
    )

    print(f"\nSplit (seed={args.seed}):")
    print(f"  Train:      {len(train_ids)} volumes ({100 * len(train_ids) / len(volume_ids):.1f}%)")
    print(f"  Validation: {len(val_ids)} volumes ({100 * len(val_ids) / len(volume_ids):.1f}%)")
    print(f"  Test:       {len(test_ids)} volumes ({100 * len(test_ids) / len(volume_ids):.1f}%)")

    print(f"\nSaving split files to: {output_dir}")
    save_splits(train_ids, val_ids, test_ids, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
