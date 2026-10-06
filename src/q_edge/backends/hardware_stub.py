"""Placeholder for running QHED on IBM Quantum hardware via Qiskit IBM Runtime.

This prototype makes no network calls. The stub documents the integration point and fails
with a clear, actionable error instead of silently returning wrong results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from q_edge.backends.base import FloatArray
from q_edge.config import QEdgeConfig


class NotConfiguredError(RuntimeError):
    """Raised when a backend needs external configuration that has not been provided."""


@dataclass(frozen=True)
class IBMHardwareBackend:
    """IBM Quantum Runtime execution using Qiskit IBM Runtime.

    Authenticates via environment variables (QISKIT_IBM_TOKEN or IBM_QUANTUM_TOKEN).
    When unconfigured, raises an informative NotConfiguredError without breaking offline usage.
    """

    shots: int = 4096
    instance: str | None = None
    name: str = "ibm-hardware"

    def run_tiles(self, batch: FloatArray) -> FloatArray:
        """Run batch of tiles on IBM Quantum hardware or raise NotConfiguredError."""
        token = os.environ.get("QISKIT_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise NotConfiguredError(
                "The IBM Quantum hardware backend is not configured. Set the 'QISKIT_IBM_TOKEN' "
                "environment variable with your API token from quantum.ibm.com. "
                "For local offline runs, use the 'numpy', 'aer', or 'aer-noisy' backends."
            )
        try:
            import importlib
            qiskit_runtime = importlib.import_module("qiskit_ibm_runtime")
            QiskitRuntimeService = qiskit_runtime.QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            backend = service.least_busy(operational=True, simulator=False)
            raise NotConfiguredError(
                f"Connected to IBM Quantum ({backend.name}), but live hardware queue execution "
                "is disabled in prototype mode to prevent queue blocking. Use 'aer' or 'numpy'."
            )
        except ImportError as err:
            raise NotConfiguredError(
                "qiskit-ibm-runtime is not installed. Install via 'pip install "
                "qiskit-ibm-runtime' to submit circuits to real IBM Quantum hardware."
            ) from err

    @classmethod
    def from_config(cls, config: QEdgeConfig) -> IBMHardwareBackend:
        """Factory used by the backend registry."""
        return cls(shots=config.shots or 4096)
