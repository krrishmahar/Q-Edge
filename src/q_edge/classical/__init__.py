"""Classical edge detectors used as baselines."""

from collections.abc import Callable

from q_edge.classical.canny import canny
from q_edge.classical.laplacian import laplacian
from q_edge.classical.prewitt import prewitt
from q_edge.classical.sobel import sobel
from q_edge.imaging import FloatArray

#: Classical detectors by name. Each maps a ``[0, 1]`` image to a ``[0, 1]`` edge map.
CLASSICAL_METHODS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "sobel": sobel,
    "prewitt": prewitt,
    "canny": canny,
    "laplacian": laplacian,
}

__all__ = ["CLASSICAL_METHODS", "canny", "laplacian", "prewitt", "sobel"]
