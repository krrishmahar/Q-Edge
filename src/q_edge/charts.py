"""Plotly figures for the Streamlit app.

Each method keeps the same colour in every chart (colour follows the entity). Hues come
from a CVD-validated categorical palette, assigned in fixed order.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from q_edge.imaging import FloatArray

#: Fixed method -> colour assignment (categorical slots 1-5 of the reference palette).
METHOD_COLORS: dict[str, str] = {
    "qhed": "#2a78d6",
    "sobel": "#eb6834",
    "prewitt": "#1baf7a",
    "canny": "#eda100",
    "laplacian": "#e87ba4",
}

METHOD_LABELS: dict[str, str] = {
    "qhed": "QHED (simulated)",
    "sobel": "Sobel",
    "prewitt": "Prewitt",
    "canny": "Canny",
    "laplacian": "Laplacian",
}

#: Diverging scale for signed differences: cool blue, neutral grey midpoint, warm orange.
DIVERGING_SCALE: list[list[float | str]] = [
    [0.0, "#2a78d6"],
    [0.5, "#d9d8d4"],
    [1.0, "#eb6834"],
]

_GRID = "rgba(128,128,128,0.18)"


def _base_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(title=x_title, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=_GRID, zeroline=False, rangemode="tozero")
    return fig


def runtime_vs_resolution(scaling: pd.DataFrame) -> go.Figure:
    """Line chart of runtime against megapixels, one line per method."""
    fig = go.Figure()
    for method, group in scaling.groupby("method", sort=False):
        name = str(method)
        group = group.sort_values("megapixels")
        label = METHOD_LABELS.get(name, name)
        fig.add_trace(
            go.Scatter(
                x=group["megapixels"],
                y=group["runtime_s"],
                mode="lines+markers",
                name=label,
                line={"color": METHOD_COLORS.get(name), "width": 2},
                marker={"size": 8},
                customdata=group[["width", "height"]],
                hovertemplate=(
                    f"<b>{label}</b><br>%{{customdata[0]}}x%{{customdata[1]}}"
                    "<br>%{x:.2f} MP<br>%{y:.3f} s<extra></extra>"
                ),
            )
        )
    return _base_layout(fig, "Runtime vs resolution", "Megapixels", "Runtime (s)")


def f1_by_method(results: pd.DataFrame) -> go.Figure:
    """Bar chart of F1 score per method."""
    methods = [str(m) for m in results["method"]]
    fig = go.Figure(
        go.Bar(
            x=[METHOD_LABELS.get(m, m) for m in methods],
            y=results["f1"],
            marker={"color": [METHOD_COLORS.get(m) for m in methods], "cornerradius": 4},
            text=[f"{v:.3f}" for v in results["f1"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>F1 %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.45)
    fig = _base_layout(fig, "F1 by method", "", "F1 score")
    fig.update_yaxes(range=[0, 1.1])
    return fig


def difference_heatmap(difference: FloatArray, classical_label: str) -> go.Figure:
    """Heatmap of ``QHED - classical`` on a diverging scale centred on zero."""
    fig = go.Figure(
        go.Heatmap(
            z=difference,
            colorscale=DIVERGING_SCALE,
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "QHED - " + classical_label, "thickness": 12},
            hovertemplate="x %{x}, y %{y}<br>difference %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title={"text": f"Difference: QHED - {classical_label}", "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
    )
    return fig
