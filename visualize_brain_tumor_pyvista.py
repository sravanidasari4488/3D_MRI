"""
High-quality volumetric brain + tumor visualization using PyVista.

Inputs:
- MRI volume: FLAIR channel from BraTS-style HDF5 slices
- Tumor segmentation mask

Rendering:
- Brain: PyVista volume rendering (add_volume) with grayscale colormap and
  CT-style opacity transfer function; smooth shading enabled
- Tumor: Red semi-transparent surface mesh overlay (marching_cubes)

Scene:
- Dark background, no axes (clean medical view)
- Realistic lighting, interactive rotation
- Opens in PyVista interactive window
"""

from __future__ import annotations

import numpy as np
from skimage import measure
import pyvista as pv

from explore_brats import load_brats_volume
from preprocess_brats_3d import load_volume_h5, normalize_volume


def ct_style_opacity(n_colors: int = 256, ramp_start: float = 0.15, ramp_end: float = 0.85):
    """
    Build an opacity array mimicking CT windowing: dark background transparent,
    soft tissue visible. Returns array of length n_colors (0–1 range for PyVista).
    """
    opacity = np.zeros(n_colors, dtype=np.float32)
    i_start = int(ramp_start * (n_colors - 1))
    i_end = int(ramp_end * (n_colors - 1))
    for i in range(i_start, min(i_end + 1, n_colors)):
        t = (i - i_start) / max(1, i_end - i_start)
        opacity[i] = 0.15 + 0.65 * t  # ramp from 0.15 to 0.8
    for i in range(i_end + 1, n_colors):
        opacity[i] = 0.8
    return opacity


def extract_surface_mesh(volume: np.ndarray, level: float) -> pv.PolyData:
    """Extract isosurface as PyVista PolyData using marching cubes."""
    vol = np.asarray(volume, dtype=np.float32)
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=level, spacing=(1.0, 1.0, 1.0)
    )
    n_faces = len(faces)
    cells = np.column_stack([np.full(n_faces, 3), faces]).ravel()
    mesh = pv.PolyData(verts, cells)
    mesh.triangulate(inplace=True)
    return mesh


def extract_tumor_surface(mask_volume: np.ndarray, level: float = 0.5) -> pv.PolyData:
    """Extract tumor surface as PyVista PolyData using marching cubes."""
    vol = mask_volume.astype(np.float32)
    if vol.max() <= level:
        raise ValueError("Tumor mask appears empty; no surface to extract.")
    return extract_surface_mesh(vol, level)


def main():
    """Load FLAIR + mask, build PyVista scene, and open interactive window."""
    print("PyVista Brain + Tumor Volumetric Visualization")
    print("==============================================")

    data_dir = input(
        "Enter path to HDF5 data directory (where volume_XXX_slice_YYY.h5 lives): "
    ).strip()
    data_dir = data_dir.strip('"').strip("'")

    volume_number = int(input("Enter volume number (e.g., 1, 2, 41): ").strip())

    print("\nLoading FLAIR volume...")
    flair = load_volume_h5(data_dir, volume_number, modality_channel=2)
    flair_norm = normalize_volume(flair)

    print("Loading ground-truth segmentation volume...")
    vols = load_brats_volume(data_dir, volume_number)
    seg = vols["Segmentation"].astype(np.uint8)
    tumor_mask = (seg > 0).astype(np.uint8)

    print("Extracting brain surface from FLAIR...")
    brain_level = float(np.percentile(flair_norm[flair_norm > 0], 15)) if np.any(flair_norm > 0) else np.percentile(flair_norm, 15)
    brain_surface = extract_surface_mesh(flair_norm, brain_level)
    print(f"  Brain mesh: {brain_surface.n_points} points, {brain_surface.n_cells} cells")

    print("Extracting tumor surface...")
    tumor_surface = extract_tumor_surface(tumor_mask, level=0.5)

    # Opacity transfer function (CT-style: background transparent, tissue visible)
    n_colors = 256
    opacity_tf = ct_style_opacity(n_colors, ramp_start=0.12, ramp_end=0.88)

    # Plotter: dark background, no axes, realistic lighting
    pl = pv.Plotter(off_screen=False)
    pl.set_background("black")

    # Brain: semi-transparent gray surface so the brain is clearly visible
    pl.add_mesh(
        brain_surface,
        color="lightgray",
        opacity=0.4,
        smooth_shading=True,
        diffuse=0.75,
        specular=0.3,
        specular_power=15.0,
    )

    # Brain: volume rendering (adds density; may be faint on some systems)
    pl.add_volume(
        flair_norm,
        cmap="gray",
        opacity=opacity_tf,
        n_colors=n_colors,
        shade=True,
        diffuse=0.7,
        specular=0.3,
        specular_power=15.0,
        show_scalar_bar=False,
    )

    # Tumor: red semi-transparent surface
    pl.add_mesh(
        tumor_surface,
        color="#cc2222",
        opacity=0.65,
        smooth_shading=True,
        diffuse=0.8,
        specular=0.4,
        specular_power=20.0,
    )

    pl.hide_axes()
    pl.view_isometric()
    pl.reset_camera()
    pl.add_title("Brain (FLAIR) + Tumor", font_size=12)

    print("\nOpening interactive window (rotate, zoom, pan with mouse)...")
    pl.show()


if __name__ == "__main__":
    main()
