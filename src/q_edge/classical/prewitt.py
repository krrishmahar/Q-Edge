"""Prewitt gradient-magnitude edge detector (OpenCV ``filter2D``)."""

from __future__ import annotations

import cv2
import numpy as np

from q_edge.imaging import FloatArray, normalise

_KX = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
_KY = _KX.T


def prewitt(image: FloatArray) -> FloatArray:
    """Return the Prewitt gradient magnitude of a ``[0, 1]`` image, normalised to ``[0, 1]``."""
    img = np.asarray(image, dtype=np.float64)
    gx = cv2.filter2D(img, cv2.CV_64F, _KX, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(img, cv2.CV_64F, _KY, borderType=cv2.BORDER_REPLICATE)
    return normalise(np.hypot(gx, gy))
