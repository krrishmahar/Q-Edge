"""Backend protocol and registry.

A backend turns a batch of ``(B, T, T)`` tiles into ``(B, T-1, T-1)`` edge magnitudes.
Adding a backend means implementing :class:`Backend` and calling :func:`register_backend`;
the pipeline and scheduler never need to change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from q_edge.config import QEdgeConfig

FloatArray = NDArray[np.float64]


@runtime_checkable
class Backend(Protocol):
    """Interface every execution backend implements.

    Implementations must be picklable so the scheduler can ship them to worker processes.
    """

    @property
    def name(self) -> str:
        """Registry name of the backend."""
        ...

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return ``(B, T-1, T-1)`` edge magnitudes for a ``(B, T, T)`` batch of tiles."""
        ...


BackendFactory = Callable[[QEdgeConfig], Backend]

_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory, *, overwrite: bool = False) -> None:
    """Register a backend factory under ``name``.

    Raises:
        ValueError: If the name is already taken and ``overwrite`` is false.
    """
    key = name.lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"backend {name!r} is already registered")
    _REGISTRY[key] = factory


def available_backends() -> list[str]:
    """Names of all registered backends."""
    return sorted(_REGISTRY)


def get_backend(name: str, config: QEdgeConfig | None = None) -> Backend:
    """Instantiate the backend registered under ``name``.

    Raises:
        ValueError: If no backend with that name exists.
    """
    factory = _REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(available_backends())}")
    return factory(config or QEdgeConfig(backend=name))
