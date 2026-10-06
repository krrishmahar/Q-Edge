"""Small image helpers shared by the quantum and classical paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Peaks below this are floating-point noise (intensities are in ``[0, 1]``), not edges.
NOISE_FLOOR = 1e-9


def normalise(magnitude: NDArray[np.floating]) -> FloatArray:
    """Scale an edge magnitude map to ``[0, 1]`` by its maximum.

    Maps whose peak is below :data:`NOISE_FLOOR` (for example rounding residue on a constant
    image) become all zeros instead of having that noise amplified to full scale.
    """
    arr = np.asarray(magnitude, dtype=np.float64)
    peak = float(np.max(arr)) if arr.size else 0.0
    return arr / peak if peak > NOISE_FLOOR else np.zeros_like(arr)


def apply_threshold(
    edges: NDArray[np.floating],
    threshold: float = 0.2,
    adaptive: bool = False,
    percentile: float = 85.0,
) -> NDArray[np.bool_]:
    """Apply classical thresholding (fixed or percentile-based adaptive) to an edge map.

    Args:
        edges: Normalised edge map with values in [0, 1].
        threshold: Fixed cutoff threshold in [0, 1].
        adaptive: If True, uses percentile thresholding on non-zero gradient magnitudes.
        percentile: Percentile cutoff (0-100) when adaptive=True.

    Returns:
        Boolean mask of detected edge pixels.
    """
    arr = np.asarray(edges, dtype=np.float64)
    if adaptive:
        nonzero = arr[arr > NOISE_FLOOR]
        if nonzero.size == 0:
            return np.zeros(arr.shape, dtype=bool)
        cutoff = float(np.percentile(nonzero, percentile))
        return arr >= max(cutoff, NOISE_FLOOR)
    return arr >= max(threshold, NOISE_FLOOR)
