"""Canny edge detector (OpenCV) with automatic hysteresis thresholds."""

from __future__ import annotations

import cv2
import numpy as np

from q_edge.imaging import FloatArray


def auto_thresholds(image_u8: np.ndarray, sigma: float = 0.33) -> tuple[float, float]:
    """Median-based hysteresis thresholds, a common parameter-free Canny heuristic.

    A floor keeps very dark images from turning sensor noise into edges.
    """
    median = float(np.median(image_u8))
    low = max(10.0, (1.0 - sigma) * median)
    high = min(255.0, max(2.0 * low, (1.0 + sigma) * median))
    return low, high


def canny(
    image: FloatArray,
    low: float | None = None,
    high: float | None = None,
    blur: int = 3,
) -> FloatArray:
    """Return a binary Canny edge map (as ``0.0`` / ``1.0`` floats).

    Args:
        image: Grayscale image in ``[0, 1]``.
        low: Lower hysteresis threshold on the 0-255 scale; automatic when ``None``.
        high: Upper hysteresis threshold on the 0-255 scale; automatic when ``None``.
        blur: Odd Gaussian pre-blur kernel size; ``0`` or ``1`` disables blurring.
    """
    img = np.clip(np.asarray(image, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    if blur > 1:
        img = np.asarray(cv2.GaussianBlur(img, (blur, blur), 0), dtype=np.uint8)
    auto_low, auto_high = auto_thresholds(img)
    edges = cv2.Canny(
        img,
        auto_low if low is None else low,
        auto_high if high is None else high,
    )
    result: FloatArray = (edges > 0).astype(np.float64)
    return result
