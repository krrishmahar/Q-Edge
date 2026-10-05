"""Placeholder for running QHED on IBM Quantum hardware via Qiskit IBM Runtime.

This prototype makes no network calls. The stub documents the integration point and fails
with a clear, actionable error instead of silently returning wrong results.
"""

from __future__ import annotations

from dataclasses import dataclass

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig


class NotConfiguredError(RuntimeError):
    """Raised when a backend needs external configuration that has not been provided."""


@dataclass(frozen=True)
class IBMHardwareBackend:
    """Interface placeholder for IBM Quantum Runtime execution.

    A real implementation would transpile each tile's QHED circuit for the target device,
    submit batches through ``qiskit_ibm_runtime.SamplerV2`` inside a ``Batch`` session, and
    convert the returned quasi-probabilities into neighbour differences, exactly as the
    shot-based Aer path does.
    """

    shots: int = 4096
    name: str = "ibm-hardware"

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Always raises :class:`NotConfiguredError`."""
        raise NotConfiguredError(
            "The IBM Quantum hardware backend is not configured. This prototype does not "
            "ship credentials or make network calls. To enable it, install "
            "qiskit-ibm-runtime, save an IBM Quantum account locally, and implement "
            "IBMHardwareBackend.run_tiles. Use the 'numpy' or 'aer' backend meanwhile."
        )

    @classmethod
    def from_config(cls, config: QEdgeConfig) -> IBMHardwareBackend:
        """Factory used by the backend registry."""
        return cls(shots=config.shots or 4096)
