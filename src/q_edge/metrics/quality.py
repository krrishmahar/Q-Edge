"""Edge-map quality metrics against a reference edge map."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EdgeScores:
    """Precision, recall and F1 of a binary edge map against a reference."""

    precision: float
    recall: float
    f1: float


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def ssim(prediction: NDArray[np.floating], reference: NDArray[np.floating]) -> float:
    """Structural similarity of two ``[0, 1]`` maps.

    Images smaller than the 7x7 SSIM window use the largest valid odd window.
    """
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    _check_shapes(a, b)
    win = min(7, *a.shape)
    win -= 1 - win % 2
    if win < 3:
        return 1.0 if np.allclose(a, b) else 0.0
    score = structural_similarity(a, b, data_range=1.0, win_size=win)  # type: ignore[no-untyped-call]
    return float(score)


def psnr(prediction: NDArray[np.floating], reference: NDArray[np.floating]) -> float:
    """Peak signal-to-noise ratio in dB for ``[0, 1]`` maps; ``inf`` for identical maps."""
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    _check_shapes(a, b)
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def edge_density(binary: NDArray[np.bool_]) -> float:
    """Fraction of pixels marked as edges."""
    arr = np.asarray(binary, dtype=bool)
    return float(arr.mean()) if arr.size else 0.0


def _within(mask: BoolArray, target: BoolArray, tolerance: float) -> int:
    """Count ``mask`` pixels lying within ``tolerance`` pixels of any ``target`` pixel."""
    if not target.any():
        return 0
    distance = distance_transform_edt(~target)
    return int(np.count_nonzero(mask & (distance <= tolerance)))


def edge_scores(
    prediction: NDArray[np.bool_], reference: NDArray[np.bool_], tolerance: float = 1.0
) -> EdgeScores:
    """Precision, recall and F1 with a pixel tolerance.

    A predicted edge pixel is a true positive if a reference edge lies within ``tolerance``
    pixels (Euclidean); a reference edge is recalled if a predicted edge lies within the
    same distance. Two empty maps score a perfect 1.0.
    """
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    _check_shapes(pred, ref)
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    if n_pred == 0 and n_ref == 0:
        return EdgeScores(1.0, 1.0, 1.0)
    precision = _within(pred, ref, tolerance) / n_pred if n_pred else 0.0
    recall = _within(ref, pred, tolerance) / n_ref if n_ref else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return EdgeScores(precision, recall, f1)
