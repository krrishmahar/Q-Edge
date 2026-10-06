"""Execution backends implementing the tile-processing protocol."""

from q_edge.backends.aer_backend import AerBackend
from q_edge.backends.base import (
    Backend,
    available_backends,
    get_backend,
    register_backend,
)
from q_edge.backends.gpu_backend import GpuBackend
from q_edge.backends.hardware_stub import IBMHardwareBackend, NotConfiguredError
from q_edge.backends.numpy_backend import NumpyBackend

register_backend("numpy", NumpyBackend.from_config)
register_backend("aer", AerBackend.from_config)
register_backend("aer-noisy", lambda cfg: AerBackend.from_config(cfg, noisy=True))
register_backend("gpu", GpuBackend.from_config)
register_backend("ibm-hardware", IBMHardwareBackend.from_config)

__all__ = [
    "AerBackend",
    "Backend",
    "GpuBackend",
    "IBMHardwareBackend",
    "NotConfiguredError",
    "NumpyBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]
