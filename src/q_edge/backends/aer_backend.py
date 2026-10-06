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
        noisy: Whether to enable a depolarizing + readout noise model.
    """

    shots: int | None = None
    seed: int | None = None
    noisy: bool = False
    name: str = "aer"

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Return QHED edge magnitudes computed from Aer circuit runs."""
        from q_edge.quantum.qhed_circuit import build_noise_model

        noise_model = build_noise_model() if self.noisy else None
        return circuit_edge_magnitude(
            batch, shots=self.shots, seed=self.seed, noise_model=noise_model
        )

    @classmethod
    def from_config(cls, config: QEdgeConfig, noisy: bool = False) -> AerBackend:
        """Factory used by the backend registry."""
        is_noisy = noisy or ("noisy" in config.backend.lower())
        return cls(
            shots=config.shots,
            seed=config.seed,
            noisy=is_noisy,
            name="aer-noisy" if is_noisy else "aer",
        )
