"""Laplacian (second-derivative) edge detector (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from q_edge.imaging import FloatArray, normalise


def laplacian(image: FloatArray, ksize: int = 3) -> FloatArray:
    """Return the absolute Laplacian response of a ``[0, 1]`` image, normalised to ``[0, 1]``."""
    img = np.asarray(image, dtype=np.float64)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    return normalise(np.abs(lap))
