"""Optional GPU backend using CuPy, with a graceful fallback to NumPy.

CuPy is not a project dependency. When it is missing, or no CUDA device is available, the
backend logs a warning and runs the identical NumPy computation instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig
from q_edge.quantum.encoding import ZERO_NORM_EPS
from q_edge.quantum.qhed_fast import AMPLITUDE_TO_DIFFERENCE

logger = logging.getLogger(__name__)


def load_cupy() -> ModuleType | None:
    """Return the ``cupy`` module if it imports and sees at least one CUDA device."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception:  # ImportError, CUDA driver/runtime errors
        return None
    module: ModuleType = cupy
    return module


def qhed_edge_magnitude_xp(tiles: Any, xp: ModuleType) -> Any:
    """QHED edge magnitude written against an array module (``numpy`` or ``cupy``).

    Mirrors :func:`q_edge.quantum.qhed_fast.qhed_edge_magnitude` step for step so it can run
    on any NumPy-compatible device.
    """
    batch = xp.asarray(tiles, dtype=xp.float64)
    b, t, _ = batch.shape
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    def differences(flat: Any) -> Any:
        norms = xp.linalg.norm(flat, axis=1)
        nonzero = norms > ZERO_NORM_EPS
        amps = xp.where(nonzero[:, None], flat / xp.where(nonzero, norms, 1.0)[:, None], 0.0)
        psi = xp.roll(xp.repeat(amps, 2, axis=1) * inv_sqrt2, -1, axis=1)
        odd = (psi[:, 0::2] - psi[:, 1::2]) * inv_sqrt2
        return odd * (AMPLITUDE_TO_DIFFERENCE * xp.where(nonzero, norms, 0.0)[:, None])

    h = differences(batch.reshape(b, t * t)).reshape(b, t, t)[:, : t - 1, : t - 1]
    transposed = xp.ascontiguousarray(batch.transpose(0, 2, 1)).reshape(b, t * t)
    v = differences(transposed).reshape(b, t, t)[:, : t - 1, : t - 1].transpose(0, 2, 1)
    return xp.hypot(h, v)


@dataclass
class GpuBackend:
    """CuPy-accelerated exact simulation; falls back to NumPy when CUDA is unavailable."""

    name: str = "gpu"
    _xp: ModuleType | None = field(default=None, init=False, repr=False)
    _checked: bool = field(default=False, init=False, repr=False)

    @property
    def uses_gpu(self) -> bool:
        """Whether a CUDA device is actually being used."""
        return self._array_module() is not np

    def _array_module(self) -> ModuleType:
        if not self._checked:
            self._xp = load_cupy()
            self._checked = True
            if self._xp is None:
                logger.warning("CuPy/CUDA not available; GPU backend is falling back to NumPy.")
        return self._xp if self._xp is not None else np

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return QHED edge magnitudes, computed on the GPU when possible."""
        xp = self._array_module()
        out = qhed_edge_magnitude_xp(batch, xp)
        result: FloatArray = np.asarray(out.get() if xp is not np else out, dtype=np.float64)
        return result

    def __getstate__(self) -> dict[str, Any]:
        # Module objects are not picklable; workers re-detect the device.
        return {"name": self.name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.name = state["name"]
        self._xp = None
        self._checked = False

    @classmethod
    def from_config(cls, config: QEdgeConfig) -> GpuBackend:
        """Factory used by the backend registry."""
        del config
        return cls()
