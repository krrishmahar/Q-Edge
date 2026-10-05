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
