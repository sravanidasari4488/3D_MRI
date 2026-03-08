import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import nibabel as nib

from explore_brats import load_brats_volume


def visualize_tumor_3d(mask_3d: np.ndarray, spacing=(1.0, 1.0, 1.0), level=0.5, title="3D Tumor Segmentation"):
    mask_3d = np.asarray(mask_3d, dtype=np.float32)
    if mask_3d.max() <= level:
        raise ValueError("Mask appears empty or below the chosen level; no surface to visualize.")

    verts, faces, normals, values = measure.marching_cubes(mask_3d, level=level, spacing=spacing)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor((1.0, 0.0, 0.0, 0.7))  # red, semi‑transparent
    mesh.set_edgecolor("k")
    mesh.set_linewidth(0.05)
    ax.add_collection3d(mesh)

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    # Equal aspect
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_y-max_range, mid_y+max_range)
    ax.set_zlim(mid_z-max_range, mid_z+max_range)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main():
    """
    Simple CLI to visualize a 3D tumor mask either from:
    - A NIfTI file (.nii/.nii.gz), or
    - BraTS-style HDF5 slices (using the Segmentation volume).
    """
    print("3D Tumor Visualization")
    print("======================")
    print("1. Load mask from NIfTI file")
    print("2. Load ground-truth mask from BraTS HDF5")
    choice = input("Select option (1 or 2, default: 2): ").strip() or "2"

    if choice == "1":
        nii_path = input("Enter path to NIfTI mask (e.g., hybrid_prediction_volume_1.nii.gz): ").strip()
        img = nib.load(nii_path)
        mask = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        visualize_tumor_3d(mask, spacing=spacing, title=f"3D Tumor Surface (NIfTI): {nii_path}")
    else:
        data_dir = input(
            "Enter path to HDF5 data directory (where volume_XXX_slice_YYY.h5 lives): "
        ).strip()
        data_dir = data_dir.strip('"').strip("'")
        volume_number = int(input("Enter volume number (e.g., 1, 2, 41): ").strip())

        print("\nLoading BraTS segmentation volume from HDF5...")
        vols = load_brats_volume(data_dir, volume_number)
        seg_multiclass = vols["Segmentation"].astype(np.uint8)
        # Binary tumor mask: any non-zero BraTS label is tumor
        mask = (seg_multiclass > 0).astype(np.uint8)

        print(f"Segmentation volume shape: {mask.shape}")
        visualize_tumor_3d(mask, spacing=(1.0, 1.0, 1.0),
                           title=f"3D Tumor Surface (BraTS volume {volume_number})")


if __name__ == "__main__":
    main()