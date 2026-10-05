"""Amplitude encoding of image tiles.

A tile of ``2**n`` pixels is flattened in row-major order and L2-normalised so it can be
loaded as the amplitudes of an ``n``-qubit state. The norm is kept so that measured
responses can be rescaled back to the original intensity scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

FloatArray = NDArray[np.float64]

#: Tile sizes supported by the pipeline (square, power-of-two side length).
SUPPORTED_TILE_SIZES: tuple[int, ...] = (4, 8, 16)

#: Norms below this value are treated as an all-zero tile.
ZERO_NORM_EPS = 1e-12


def num_data_qubits(num_pixels: int) -> int:
    """Return the number of qubits needed to amplitude-encode ``num_pixels`` values.

    Raises:
        ValueError: If ``num_pixels`` is not a power of two greater than one.
    """
    if num_pixels < 2 or num_pixels & (num_pixels - 1):
        raise ValueError(f"number of pixels must be a power of two >= 2, got {num_pixels}")
    return num_pixels.bit_length() - 1


def normalize_tiles(tiles: NDArray[np.floating]) -> tuple[FloatArray, FloatArray]:
    """Flatten and L2-normalise a batch of tiles.

    Args:
        tiles: Array of shape ``(B, H, W)`` or ``(B, N)`` with ``H * W`` (or ``N``) a power
            of two.

    Returns:
        ``(amplitudes, norms)`` where ``amplitudes`` has shape ``(B, N)`` and unit L2 norm
        per row, and ``norms`` has shape ``(B,)``. All-zero tiles yield zero amplitudes and
        a norm of ``0`` instead of NaNs.
    """
    batch = np.asarray(tiles, dtype=np.float64)
    if batch.ndim < 2:
        raise ValueError(f"expected a batch of tiles with ndim >= 2, got shape {batch.shape}")
    flat = batch.reshape(batch.shape[0], -1)
    num_data_qubits(flat.shape[1])
    norms = np.linalg.norm(flat, axis=1)
    nonzero = norms > ZERO_NORM_EPS
    safe = np.where(nonzero, norms, 1.0)
    amplitudes = np.where(nonzero[:, None], flat / safe[:, None], 0.0)
    return amplitudes, np.where(nonzero, norms, 0.0)


def amplitude_encoding_circuit(amplitudes: NDArray[np.floating]) -> QuantumCircuit:
    """Build a circuit that prepares the given unit-norm amplitude vector.

    Args:
        amplitudes: 1-D array of length ``2**n`` with unit L2 norm.

    Raises:
        ValueError: If the vector is not normalised (for example an all-zero tile).
    """
    vec = np.asarray(amplitudes, dtype=np.float64).ravel()
    n = num_data_qubits(vec.size)
    if not np.isclose(np.linalg.norm(vec), 1.0, atol=1e-9):
        raise ValueError("amplitudes must have unit L2 norm; zero tiles cannot be encoded")
    qc = QuantumCircuit(n, name="encode")
    qc.append(StatePreparation(vec.tolist()), range(n))
    return qc
