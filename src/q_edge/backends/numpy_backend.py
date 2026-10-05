"""CPU backend using the exact batched NumPy statevector simulation."""

from __future__ import annotations

from dataclasses import dataclass

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig
from q_edge.quantum.qhed_fast import qhed_edge_magnitude


@dataclass(frozen=True)
class NumpyBackend:
    """Vectorised exact simulation of QHED on the CPU. The default backend."""

    name: str = "numpy"

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return QHED edge magnitudes for a batch of tiles."""
        return qhed_edge_magnitude(batch)

    @classmethod
    def from_config(cls, config: QEdgeConfig) -> NumpyBackend:
        """Factory used by the backend registry."""
        del config
        return cls()
