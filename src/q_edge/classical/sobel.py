"""Sobel gradient-magnitude edge detector (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from q_edge.imaging import FloatArray, normalise


def sobel(image: FloatArray, ksize: int = 3) -> FloatArray:
    """Return the Sobel gradient magnitude of a ``[0, 1]`` image, normalised to ``[0, 1]``."""
    img = np.asarray(image, dtype=np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    return normalise(np.hypot(gx, gy))
