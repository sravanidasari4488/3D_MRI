"""
Advanced interactive 3D brain + tumor visualization.

Inputs:
- MRI volume: FLAIR channel from BraTS-style HDF5 slices
- Tumor mask: BraTS segmentation (any non-zero label = tumor)

Rendering (both as Mesh3d for reliable display in the browser):
- Brain: Semi-transparent gray mesh from FLAIR (marching_cubes)
- Tumor: Solid red mesh with slight gloss (marching_cubes on mask)

Features:
- Smooth lighting, camera controls, dark background, hidden axes
- Saves interactive HTML and opens in browser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import measure
import plotly.graph_objects as go
import plotly.offline as pyo

from explore_brats import load_brats_volume
from preprocess_brats_3d import load_volume_h5, normalize_volume


def extract_brain_surface_from_flair(
    flair_volume: np.ndarray, level: float | None = None
):
    """
    Extract brain surface from FLAIR volume using marching cubes.
    Uses a low intensity threshold so the brain envelope is visible.
    """
    vol = flair_volume.astype(np.float32)
    if level is None:
        nonzero = vol[vol > 0]
        level = float(np.percentile(nonzero, 15)) if nonzero.size else np.percentile(vol, 15)
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=level, spacing=(1.0, 1.0, 1.0)
    )
    return verts, faces


def extract_tumor_surface_from_mask(mask_volume: np.ndarray, level: float = 0.5):
    """
    Extract tumor surface from a 3D binary mask using marching cubes.
    """
    vol = mask_volume.astype(np.float32)
    if vol.max() <= level:
        raise ValueError(
            "Tumor mask appears empty or below threshold; no tumor surface to extract."
        )
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=level, spacing=(1.0, 1.0, 1.0)
    )
    return verts, faces


def build_volume_rendering_figure(
    brain_verts: np.ndarray,
    brain_faces: np.ndarray,
    tumor_verts: np.ndarray,
    tumor_faces: np.ndarray,
    title: str,
) -> go.Figure:
    """
    Build Plotly figure: Brain as semi-transparent gray Mesh3d, tumor as red Mesh3d.
    Dark theme, hidden axes, smooth lighting. Both meshes render reliably in the browser.
    """
    # Brain: semi-transparent gray mesh so brain and tumor are both visible
    brain_mesh = go.Mesh3d(
        x=brain_verts[:, 0],
        y=brain_verts[:, 1],
        z=brain_verts[:, 2],
        i=brain_faces[:, 0],
        j=brain_faces[:, 1],
        k=brain_faces[:, 2],
        color="rgb(180, 180, 180)",
        opacity=0.35,
        name="Brain (FLAIR)",
        lighting=dict(
            ambient=0.6,
            diffuse=0.75,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2,
        ),
        lightposition=dict(x=300, y=300, z=500),
        flatshading=False,
    )

    # Tumor: solid red mesh, slightly glossy
    tumor_mesh = go.Mesh3d(
        x=tumor_verts[:, 0],
        y=tumor_verts[:, 1],
        z=tumor_verts[:, 2],
        i=tumor_faces[:, 0],
        j=tumor_faces[:, 1],
        k=tumor_faces[:, 2],
        color="rgb(220, 50, 50)",
        opacity=0.92,
        name="Tumor",
        lighting=dict(
            ambient=0.45,
            diffuse=0.85,
            specular=0.55,
            roughness=0.35,
            fresnel=0.2,
        ),
        lightposition=dict(x=300, y=300, z=500),
        flatshading=False,
    )

    fig = go.Figure(data=[brain_mesh, tumor_mesh])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="rgb(240,240,240)"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="rgb(18, 18, 18)",
        plot_bgcolor="rgb(18, 18, 18)",
        scene=dict(
            xaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title="",
            ),
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title="",
            ),
            zaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title="",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor="rgb(18, 18, 18)",
        ),
        legend=dict(
            font=dict(color="rgb(200,200,200)", size=12),
            bgcolor="rgba(40,40,40,0.7)",
            bordercolor="rgba(100,100,100,0.5)",
            x=0.02,
            y=0.98,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def main():
    """
    CLI: load FLAIR + mask, build volume + mesh figure, save HTML, open in browser.
    """
    print("Advanced 3D Brain + Tumor Visualization (Volume Rendering)")
    print("===========================================================")

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
    seg_multiclass = vols["Segmentation"].astype(np.uint8)
    tumor_mask = (seg_multiclass > 0).astype(np.uint8)

    print("Extracting brain surface from FLAIR...")
    brain_verts, brain_faces = extract_brain_surface_from_flair(flair_norm)
    print(f"  Brain mesh: {len(brain_verts)} vertices, {len(brain_faces)} faces")

    print("Extracting tumor surface from mask...")
    tumor_verts, tumor_faces = extract_tumor_surface_from_mask(tumor_mask, level=0.5)
    print(f"  Tumor mesh: {len(tumor_verts)} vertices, {len(tumor_faces)} faces")

    title = f"Brain + Tumor (BraTS volume {volume_number}) — Gray: brain (FLAIR), Red: tumor"
    fig = build_volume_rendering_figure(
        brain_verts, brain_faces, tumor_verts, tumor_faces, title
    )

    print("Rendering: brain (gray mesh) + tumor (red mesh).")
    output_path = Path(f"brain_tumor_volume_3d_volume_{volume_number}.html")
    print(f"\nSaving interactive HTML: {output_path.resolve()}")
    pyo.plot(fig, filename=str(output_path), auto_open=True)


if __name__ == "__main__":
    main()
