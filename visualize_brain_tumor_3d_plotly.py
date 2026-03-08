"""
Interactive 3D brain + tumor visualization in a web browser using Plotly.

Inputs:
- MRI volume: FLAIR channel from BraTS-style HDF5 slices
- Tumor mask: BraTS segmentation volume (any non-zero label = tumor)

Visualization:
- Brain surface extracted from FLAIR intensity using an automatic threshold
- Tumor surface extracted from the binary mask
- Brain rendered as semi-transparent gray mesh
- Tumor rendered as solid red mesh
- Fully interactive (rotate, zoom, pan) Plotly scene saved as HTML and
  automatically opened in the default web browser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import measure
import plotly.graph_objects as go
import plotly.offline as pyo

from explore_brats import load_brats_volume
from preprocess_brats_3d import load_volume_h5, normalize_volume


def extract_brain_surface_from_flair(flair_volume: np.ndarray, level: float | None = None):
    """
    Extract brain surface from a FLAIR volume using marching cubes.

    Parameters
    ----------
    flair_volume : np.ndarray
        3D FLAIR volume (H, W, D).
    level : float or None
        Intensity threshold for marching cubes. If None, an automatic
        threshold based on the 20th percentile of non-zero intensities is used.

    Returns
    -------
    verts, faces : np.ndarray, np.ndarray
        Vertices and faces of the extracted brain surface mesh.
    """
    vol = flair_volume.astype(np.float32)

    # Choose an automatic threshold if not provided
    if level is None:
        nonzero = vol[vol > 0]
        if nonzero.size == 0:
            # Fallback: use global percentile
            level = np.percentile(vol, 20)
        else:
            level = float(np.percentile(nonzero, 20))

    verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=(1.0, 1.0, 1.0))
    return verts, faces


def extract_tumor_surface_from_mask(mask_volume: np.ndarray, level: float = 0.5):
    """
    Extract tumor surface from a 3D binary mask using marching cubes.

    Parameters
    ----------
    mask_volume : np.ndarray
        3D tumor mask (H, W, D) with values 0/1.
    level : float
        Isosurface level; 0.5 works well for binary masks.
    """
    vol = mask_volume.astype(np.float32)
    if vol.max() <= level:
        raise ValueError("Tumor mask appears empty or below threshold; no tumor surface to extract.")

    verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=(1.0, 1.0, 1.0))
    return verts, faces


def create_brain_tumor_figure(
    brain_verts: np.ndarray,
    brain_faces: np.ndarray,
    tumor_verts: np.ndarray,
    tumor_faces: np.ndarray,
    title: str,
) -> go.Figure:
    """
    Build a Plotly figure with brain and tumor meshes.
    """
    # Brain mesh (semi-transparent gray)
    brain_mesh = go.Mesh3d(
        x=brain_verts[:, 0],
        y=brain_verts[:, 1],
        z=brain_verts[:, 2],
        i=brain_faces[:, 0],
        j=brain_faces[:, 1],
        k=brain_faces[:, 2],
        color="lightgray",
        opacity=0.20,
        name="Brain",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3),
        flatshading=True,
    )

    # Tumor mesh (solid red)
    tumor_mesh = go.Mesh3d(
        x=tumor_verts[:, 0],
        y=tumor_verts[:, 1],
        z=tumor_verts[:, 2],
        i=tumor_faces[:, 0],
        j=tumor_faces[:, 1],
        k=tumor_faces[:, 2],
        color="red",
        opacity=0.8,
        name="Tumor",
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5),
        flatshading=True,
    )

    fig = go.Figure(data=[brain_mesh, tumor_mesh])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (voxels)",
            yaxis_title="Y (voxels)",
            zaxis_title="Z (voxels)",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            aspectmode="data",
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.1)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def main():
    """
    CLI to generate an interactive 3D brain + tumor visualization and
    open it in a web browser.
    """
    print("Interactive 3D Brain + Tumor Visualization (Plotly)")
    print("====================================================")

    data_dir = input(
        "Enter path to HDF5 data directory (where volume_XXX_slice_YYY.h5 lives): "
    ).strip()
    data_dir = data_dir.strip('"').strip("'")

    volume_number = int(input("Enter volume number (e.g., 1, 2, 41): ").strip())

    # Load FLAIR (for brain surface) and segmentation (for tumor)
    print("\nLoading FLAIR volume...")
    flair = load_volume_h5(data_dir, volume_number, modality_channel=2)
    flair_norm = normalize_volume(flair)

    print("Loading ground-truth segmentation volume...")
    vols = load_brats_volume(data_dir, volume_number)
    seg_multiclass = vols["Segmentation"].astype(np.uint8)
    tumor_mask = (seg_multiclass > 0).astype(np.uint8)  # any non-zero label = tumor

    print("\nExtracting brain surface from FLAIR...")
    brain_verts, brain_faces = extract_brain_surface_from_flair(flair_norm)
    print(f"  Brain mesh: {len(brain_verts)} vertices, {len(brain_faces)} faces")

    print("Extracting tumor surface from segmentation mask...")
    tumor_verts, tumor_faces = extract_tumor_surface_from_mask(tumor_mask, level=0.5)
    print(f"  Tumor mesh: {len(tumor_verts)} vertices, {len(tumor_faces)} faces")

    title = f"3D Brain + Tumor Surface (BraTS volume {volume_number})"
    fig = create_brain_tumor_figure(brain_verts, brain_faces, tumor_verts, tumor_faces, title)

    # Save as interactive HTML and open in browser
    output_path = Path(f"brain_tumor_3d_volume_{volume_number}.html")
    print(f"\nSaving interactive visualization to: {output_path.resolve()}")
    pyo.plot(fig, filename=str(output_path), auto_open=True)


if __name__ == "__main__":
    main()

