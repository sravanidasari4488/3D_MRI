# Brain Tumor Segmentation

A 2D U-Net pipeline for brain tumor segmentation on BraTS-style MRI data, with slice-based loading, class balancing, and multi-class Dice evaluation.

---

## Dataset

**You do not have the dataset in this folder.** Download the BraTS 2020 dataset from Kaggle:

- **BraTS 2020 Training Data**: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

After downloading, extract the archive so that slice files (`volume_XXX_slice_YYY.h5`) are in a directory accessible to the scripts. Each `.h5` file contains:
- `image`: (240, 240, 4) — T1, T2, FLAIR, T1CE
- `mask`: (240, 240, 3) — segmentation labels (NCR, ED, ET)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Pipeline

### Option A: Jupyter Notebook (recommended for GitHub)

Run the full pipeline in one notebook. No large files are stored in the repo—models and data stay local.

```bash
jupyter notebook training/brain_tumor_training_pipeline.ipynb
```

Edit the config cell to set `DATA_DIR` to your BraTS 2020 data path, then run all cells.

### Option B: Command-line scripts

1. **Split dataset** — Divide volume IDs into train/val/test (70/15/15):
   ```bash
   python split_brats_dataset.py <data_dir> -o splits
   ```

2. **Train 2D U-Net** — Train on balanced slices:
   ```bash
   python -m training.train_2d_unet
   ```
   Prompts for splits directory and HDF5 data path. Saves `best_model.h5`.

3. **Evaluate** — Run per-volume Dice on the test set:
   ```bash
   python -m training.evaluate_2d
   ```

---

## Project Structure

```
MRI/
├── training/
│   ├── brain_tumor_training_pipeline.ipynb  # Full pipeline (split→train→evaluate)
│   ├── dataset_loader_2d.py   # Load and balance 2D slices
│   ├── unet2d_model.py        # 2D U-Net (input 240×240×4, output 3 classes)
│   ├── train_2d_unet.py       # Training script
│   └── evaluate_2d.py         # Per-volume Dice evaluation
├── splits/
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── test_ids.txt
├── split_brats_dataset.py     # Volume-level train/val/test split
├── explore_brats.py           # Explore BraTS HDF5 structure
├── ggmm_segmentation.py       # GGMM-based segmentation
├── evaluate_segmentation.py   # GGMM evaluation (no U-Net)
└── requirements.txt
```

---

## Notes

- Ensure the HDF5 data directory contains files matching `volume_<N>_slice_<K>.h5`, or a `data/` subdirectory with that structure.
- The 2D U-Net uses a combined loss (Dice + Categorical Crossentropy) and outputs 3-class softmax predictions.
