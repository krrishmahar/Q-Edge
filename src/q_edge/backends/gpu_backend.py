"""CUDA backend using CuPy.

CuPy remains optional, but selecting this backend never disguises CPU execution as GPU
execution. Capability probing is separate from backend execution so the UI can disable the
option before a request is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig
from q_edge.quantum.encoding import ZERO_NORM_EPS
from q_edge.quantum.qhed_fast import AMPLITUDE_TO_DIFFERENCE


@dataclass(frozen=True)
class GpuStatus:
    """Observable CUDA capability information for the current Python environment."""

    available: bool
    usable: bool
    name: str | None = None
    compute_capability: str | None = None
    memory_total_mb: float | None = None
    cupy_version: str | None = None
    reason: str | None = None


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


def gpu_status() -> GpuStatus:
    """Probe CUDA and run a tiny CuPy kernel before reporting the GPU as usable."""
    try:
        import cupy
    except ImportError:
        return GpuStatus(False, False, reason="CuPy is not installed.")

    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return GpuStatus(
                False,
                False,
                cupy_version=cupy.__version__,
                reason="No CUDA device was found.",
            )
        device = cupy.cuda.Device(0)
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        probe = cupy.array([1], dtype=cupy.int32) + 1
        probe.item()
        device.synchronize()
        total = cupy.cuda.Device(0).mem_info[1] / 1024**2
        return GpuStatus(
            True,
            True,
            name=str(name),
            compute_capability=str(device.compute_capability),
            memory_total_mb=round(total, 1),
            cupy_version=cupy.__version__,
        )
    except Exception as exc:
        return GpuStatus(
            True,
            False,
            cupy_version=cupy.__version__,
            reason=f"CUDA is visible but a CuPy kernel failed: {exc}",
        )


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
    """CuPy-accelerated exact simulation that requires a usable CUDA device."""

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
                status = gpu_status()
                raise RuntimeError(status.reason or "CUDA GPU is unavailable.")
        return self._xp

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return QHED edge magnitudes, computed on the GPU when possible."""
        xp = self._array_module()
        out = qhed_edge_magnitude_xp(batch, xp)
        result: FloatArray = np.asarray(out.get(), dtype=np.float64)
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
