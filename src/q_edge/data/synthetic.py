"""Deterministic synthetic test images with known ground-truth edges."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def label_edges(labels: NDArray[np.integer]) -> BoolArray:
    """Mark pixels whose right or lower neighbour belongs to a different region."""
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return edges


def synthetic_shapes(height: int = 256, width: int = 256) -> tuple[FloatArray, BoolArray]:
    """Render piecewise-constant shapes on a dark background.

    The layout scales with the image size, so the same scene can be rendered at any
    resolution for scaling benchmarks.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(image, edges)``: a float image in ``[0, 1]`` and a boolean ground-truth edge map.
    """
    if height < 8 or width < 8:
        raise ValueError("synthetic images must be at least 8x8")
    yy, xx = np.mgrid[0:height, 0:width]
    u, v = xx / width, yy / height
    labels = np.zeros((height, width), dtype=np.int32)
    labels[(u > 0.08) & (u < 0.42) & (v > 0.1) & (v < 0.55)] = 1  # rectangle
    labels[(u - 0.68) ** 2 + ((v - 0.35) * height / width) ** 2 < 0.18**2] = 2  # disc
    labels[(v > 0.65) & (v < 0.9) & (np.abs(u - 0.5) < (v - 0.65) * 1.2)] = 3  # triangle
    labels[(u > 0.12) & (u < 0.3) & (v > 0.7) & (v < 0.85)] = 4  # small square
    intensity = np.array([0.1, 0.85, 0.55, 0.7, 0.35])
    image: FloatArray = intensity[labels]
    return image, label_edges(labels)
