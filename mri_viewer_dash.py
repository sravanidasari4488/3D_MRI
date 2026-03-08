"""
Professional web-based MRI viewer using Dash + Plotly.

Features:
- Axial, Coronal, Sagittal slice viewers with crosshairs
- Sliders: slice index (per plane), opacity, intensity threshold
- 3D volume rendering (go.Volume) for brain with tumor overlay in red
- WebGL-accelerated 3D, medical-style black UI
- Load BraTS volume from HDF5 (data dir + volume number)

Run: python mri_viewer_dash.py
Then open http://127.0.0.1:8050/ in browser.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash

# In-memory cache for loaded volume/mask (avoids sending large arrays to client)
VOLUME_CACHE = {
    "flair": None,   # 3D float, normalized FLAIR
    "mask": None,    # 3D uint8, binary tumor
    "shape": None,   # (H, W, D)
}

# Grayscale colorscale for slice views (visible on black background)
GRAYSCALE_CMAP = [[0, "black"], [0.5, "gray"], [1, "white"]]

def load_volume_from_brats(data_dir: str, volume_number: int):
    """Load FLAIR and binary tumor mask; store in VOLUME_CACHE."""
    from preprocess_brats_3d import load_volume_h5, normalize_volume
    from explore_brats import load_brats_volume

    data_dir = data_dir.strip('"').strip("'")
    flair = load_volume_h5(data_dir, volume_number, modality_channel=2)
    flair_norm = normalize_volume(flair)

    vols = load_brats_volume(data_dir, volume_number)
    seg = vols["Segmentation"].astype(np.uint8)
    mask = (seg > 0).astype(np.uint8)

    VOLUME_CACHE["flair"] = flair_norm
    VOLUME_CACHE["mask"] = mask
    VOLUME_CACHE["shape"] = flair_norm.shape
    return flair_norm.shape


def make_slice_figure(volume_2d, title, crosshair_x=None, crosshair_y=None,
                      x_label="X", y_label="Y", aspect_ratio=1.0):
    """Build a single slice plot with optional crosshair lines (medical style, black bg)."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=volume_2d,
        colorscale=GRAYSCALE_CMAP,
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white", size=12),
        xaxis=dict(
            title=x_label,
            showgrid=False,
            zeroline=False,
            linecolor="gray",
            tickfont=dict(color="white"),
            title_font=dict(color="white"),
        ),
        yaxis=dict(
            title=y_label,
            showgrid=False,
            zeroline=False,
            linecolor="gray",
            scaleanchor="x",
            scaleratio=aspect_ratio,
            tickfont=dict(color="white"),
            title_font=dict(color="white"),
        ),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    shapes = []
    if crosshair_x is not None:
        shapes.append(dict(
            type="line",
            x0=crosshair_x, x1=crosshair_x,
            y0=0, y1=volume_2d.shape[0],
            line=dict(color="cyan", width=1, dash="dot"),
        ))
    if crosshair_y is not None:
        shapes.append(dict(
            type="line",
            x0=0, x1=volume_2d.shape[1],
            y0=crosshair_y, y1=crosshair_y,
            line=dict(color="cyan", width=1, dash="dot"),
        ))
    if shapes:
        fig.update_layout(shapes=shapes)
    return fig


def make_volume_figure(flair_3d, mask_3d, opacity=0.3, intensity_threshold_pct=20):
    """Build 3D volume plot: brain (grayscale) + tumor (red overlay), WebGL."""
    H, W, D = flair_3d.shape
    # go.Volume: value is 3D with shape (len(x), len(y), len(z)); value[i,j,k] at (x[i], y[j], z[k])
    # Our volume is [H, W, D] = rows, cols, depth. Use x=cols(W), y=rows(H), z=depth(D)
    x = np.arange(W, dtype=np.int32)
    y = np.arange(H, dtype=np.int32)
    z = np.arange(D, dtype=np.int32)
    values_brain = np.transpose(flair_3d, (1, 0, 2))  # (W, H, D)
    values_mask = np.transpose(mask_3d.astype(np.float32), (1, 0, 2))  # (W, H, D)

    vmin = float(np.percentile(flair_3d[flair_3d > 0], intensity_threshold_pct)) if (flair_3d > 0).any() else float(flair_3d.min())
    vmax = float(flair_3d.max())

    fig = go.Figure()

    # Brain volume (grayscale, semi-transparent)
    fig.add_trace(go.Volume(
        x=x,
        y=y,
        z=z,
        value=values_brain,
        isomin=vmin,
        isomax=vmax,
        surface_count=21,
        opacity=opacity,
        colorscale=[[0, "black"], [0.5, "gray"], [1, "white"]],
        showscale=False,
        name="Brain",
    ))

    # Tumor overlay (red where mask > 0)
    if values_mask.max() > 0:
        fig.add_trace(go.Volume(
            x=x,
            y=y,
            z=z,
            value=values_mask,
            isomin=0.5,
            isomax=1,
            surface_count=1,
            opacity=0.9,
            colorscale=[[0, "rgba(0,0,0,0)"], [0.5, "rgba(0,0,0,0)"], [1, "red"]],
            showscale=False,
            name="Tumor",
        ))

    fig.update_layout(
        title="3D Volume (Brain + Tumor)",
        paper_bgcolor="black",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(backgroundcolor="black", gridcolor="gray", title="X"),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", title="Y"),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


# Build app
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "MRI Viewer"

# Layout: black theme, sidebar + main
app.layout = html.Div([
    html.Div([
        html.H2("MRI Viewer", style=dict(color="white")),
        html.Hr(style=dict(borderColor="gray")),
        html.Label("Data directory", style=dict(color="white")),
        dcc.Input(id="data-dir", type="text", placeholder="path/to/h5/data", style=dict(width="100%", marginBottom="8px")),
        html.Label("Volume number", style=dict(color="white")),
        dcc.Input(id="volume-num", type="number", value=1, min=1, style=dict(width="100%", marginBottom="12px")),
        html.Button("Load volume", id="btn-load", n_clicks=0, style=dict(marginBottom="20px")),
        html.Div(id="load-status", style=dict(color="lime", fontSize="12px", marginBottom="16px")),

        html.Label("Axial slice", style=dict(color="white")),
        dcc.Slider(id="slider-axial", min=0, max=1, value=0, step=1, marks=None),
        html.Label("Coronal slice", style=dict(color="white")),
        dcc.Slider(id="slider-coronal", min=0, max=1, value=0, step=1, marks=None),
        html.Label("Sagittal slice", style=dict(color="white")),
        dcc.Slider(id="slider-sagittal", min=0, max=1, value=0, step=1, marks=None),
        html.Label("Opacity (3D)", style=dict(color="white")),
        dcc.Slider(id="slider-opacity", min=0.1, max=1, value=0.3, step=0.05),
        html.Label("Intensity threshold % (3D)", style=dict(color="white")),
        dcc.Slider(id="slider-threshold", min=0, max=80, value=20, step=5),
    ], style=dict(
        position="fixed", left=0, top=0, bottom=0, width="260px",
        padding="16px", backgroundColor="#0d0d0d", overflowY="auto",
        borderRight="1px solid #333",
    )),

    html.Div([
        html.Div([
            html.Div([dcc.Graph(id="graph-axial", config=dict(scrollZoom=True))], style=dict(flex=1, minWidth="280px")),
            html.Div([dcc.Graph(id="graph-coronal", config=dict(scrollZoom=True))], style=dict(flex=1, minWidth="280px")),
            html.Div([dcc.Graph(id="graph-sagittal", config=dict(scrollZoom=True))], style=dict(flex=1, minWidth="280px")),
        ], style=dict(display="flex", gap="12px", marginBottom="12px", flexWrap="wrap")),
        html.Div([
            dcc.Graph(id="graph-volume", config=dict(scrollZoom=True)),
        ], style=dict(height="420px")),
    ], style=dict(marginLeft="276px", padding="16px", backgroundColor="black", minHeight="100vh")),
], style=dict(backgroundColor="black"))


@app.callback(
    [Output("load-status", "children"),
     Output("slider-axial", "max"),
     Output("slider-coronal", "max"),
     Output("slider-sagittal", "max"),
     Output("slider-axial", "value"),
     Output("slider-coronal", "value"),
     Output("slider-sagittal", "value")],
    Input("btn-load", "n_clicks"),
    [State("data-dir", "value"),
     State("volume-num", "value")],
    prevent_initial_call=True,
)
def on_load_volume(n_clicks, data_dir, volume_num):
    if not data_dir or volume_num is None:
        return "Enter data dir and volume number.", 1, 1, 1, 0, 0, 0
    try:
        shape = load_volume_from_brats(data_dir, int(volume_num))
        H, W, D = shape
        return (
            f"Loaded: {H}×{W}×{D}",
            max(0, D - 1),
            max(0, W - 1),
            max(0, H - 1),
            D // 2,
            W // 2,
            H // 2,
        )
    except Exception as e:
        return f"Error: {str(e)}", 1, 1, 1, 0, 0, 0


@app.callback(
    [Output("graph-axial", "figure"),
     Output("graph-coronal", "figure"),
     Output("graph-sagittal", "figure")],
    [Input("slider-axial", "value"),
     Input("slider-coronal", "value"),
     Input("slider-sagittal", "value")],
)
def update_slices(k_axial, j_coronal, i_sagittal):
    flair = VOLUME_CACHE["flair"]
    mask = VOLUME_CACHE["mask"]
    if flair is None or mask is None:
        empty = go.Figure().add_annotation(text="Load a volume first", x=0.5, y=0.5, showarrow=False)
        empty.update_layout(paper_bgcolor="black", font=dict(color="white"))
        return empty, empty, empty

    H, W, D = flair.shape
    k_axial = min(max(0, int(k_axial)), D - 1)
    j_coronal = min(max(0, int(j_coronal)), W - 1)
    i_sagittal = min(max(0, int(i_sagittal)), H - 1)

    # Axial: slice at z = k_axial; crosshair at sagittal x = i_sagittal, coronal y = j_coronal
    axial_slice = flair[:, :, k_axial]
    fig_axial = make_slice_figure(
        axial_slice,
        f"Axial (slice {k_axial})",
        crosshair_x=i_sagittal,
        crosshair_y=j_coronal,
        x_label="X",
        y_label="Y",
    )

    # Coronal: slice at y = j_coronal; crosshair at x = i_sagittal, z = k_axial
    coronal_slice = flair[:, j_coronal, :]  # (H, D)
    fig_coronal = make_slice_figure(
        coronal_slice,
        f"Coronal (slice {j_coronal})",
        crosshair_x=i_sagittal,
        crosshair_y=k_axial,
        x_label="X",
        y_label="Z",
    )

    # Sagittal: slice at x = i_sagittal; crosshair at y = j_coronal, z = k_axial
    sagittal_slice = flair[i_sagittal, :, :]  # (W, D)
    fig_sagittal = make_slice_figure(
        sagittal_slice,
        f"Sagittal (slice {i_sagittal})",
        crosshair_x=j_coronal,
        crosshair_y=k_axial,
        x_label="Y",
        y_label="Z",
    )

    return fig_axial, fig_coronal, fig_sagittal


@app.callback(
    Output("graph-volume", "figure"),
    [Input("slider-opacity", "value"),
     Input("slider-threshold", "value")],
)
def update_volume(opacity, threshold_pct):
    flair = VOLUME_CACHE["flair"]
    mask = VOLUME_CACHE["mask"]
    if flair is None or mask is None:
        fig = go.Figure()
        fig.add_annotation(text="Load a volume first", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="black", font=dict(color="white"))
        return fig
    return make_volume_figure(flair, mask, opacity=float(opacity), intensity_threshold_pct=int(threshold_pct))


if __name__ == "__main__":
    print("MRI Viewer starting at http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)
