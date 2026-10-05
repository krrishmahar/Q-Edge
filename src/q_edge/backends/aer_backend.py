"""Qiskit Aer backend: runs one real QHED circuit per tile and orientation."""

from __future__ import annotations

from dataclasses import dataclass

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig
from q_edge.quantum.qhed_circuit import circuit_edge_magnitude


@dataclass(frozen=True)
class AerBackend:
    """Circuit-level simulation on Qiskit Aer.

    Attributes:
        shots: ``None`` for exact statevector simulation, otherwise shots per circuit.
        seed: Seed for shot sampling.

    This backend builds and simulates ``2 * B`` circuits per batch. It is orders of
    magnitude slower than :class:`~q_edge.backends.numpy_backend.NumpyBackend` and is
    intended for small images and validation.
    """

    shots: int | None = None
    seed: int | None = None
    name: str = "aer"

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return QHED edge magnitudes computed from Aer circuit runs."""
        return circuit_edge_magnitude(batch, shots=self.shots, seed=self.seed)

    @classmethod
    def from_config(cls, config: QEdgeConfig) -> AerBackend:
        """Factory used by the backend registry."""
        return cls(shots=config.shots, seed=config.seed)
