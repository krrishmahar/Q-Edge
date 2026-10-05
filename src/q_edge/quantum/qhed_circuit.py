"""QHED as real Qiskit circuits, executed on Qiskit Aer.

The circuit for one tile uses ``n`` data qubits plus one ancilla (qubit 0, the
least-significant bit), so its size is bounded by the tile, never by the image:

``encode(c) on data`` -> ``H(anc)`` -> ``decrement on all n + 1 qubits`` -> ``H(anc)``

This module is the reference implementation used to prove that
:mod:`q_edge.quantum.qhed_fast` is faithful. It is far slower and is meant for small
images, validation and, later, hardware execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from q_edge.quantum.encoding import (
    FloatArray,
    amplitude_encoding_circuit,
    normalize_tiles,
    num_data_qubits,
)
from q_edge.quantum.qhed_fast import AMPLITUDE_TO_DIFFERENCE, combine, oriented_responses


def decrement_circuit(num_qubits: int) -> QuantumCircuit:
    """Return the amplitude permutation ``|k> -> |k - 1 mod 2**num_qubits>``.

    Built from multi-controlled X gates: bit ``j`` flips when all lower bits are ``0``.
    Gates run from the lowest bit upwards so each control sees the already-updated bits.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    qc = QuantumCircuit(num_qubits, name="dec")
    qc.x(0)
    for target in range(1, num_qubits):
        controls = list(range(target))
        # After the lower bits were decremented, they are all 1 exactly when they were all 0.
        qc.mcx(controls, target)
    return qc


def build_qhed_circuit(amplitudes: NDArray[np.floating], measure: bool = False) -> QuantumCircuit:
    """Build the QHED circuit for one normalised, flattened tile.

    Args:
        amplitudes: Unit-norm vector of length ``2**n``.
        measure: Append measurements of all qubits (for shot-based execution).

    Returns:
        A circuit on ``n + 1`` qubits; qubit 0 is the ancilla.
    """
    vec = np.asarray(amplitudes, dtype=np.float64).ravel()
    n = num_data_qubits(vec.size)
    qc = QuantumCircuit(n + 1, name="qhed")
    qc.compose(amplitude_encoding_circuit(vec), qubits=range(1, n + 1), inplace=True)
    qc.h(0)
    qc.compose(decrement_circuit(n + 1), qubits=range(n + 1), inplace=True)
    qc.h(0)
    if measure:
        qc.measure_all()
    return qc


def _run(circuits: Sequence[QuantumCircuit], shots: int | None, seed: int | None) -> Any:
    """Transpile and run circuits on a statevector-method Aer simulator."""
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    compiled = transpile(list(circuits), sim, optimization_level=0)
    return sim.run(compiled, shots=1 if shots is None else shots).result()


def _real_up_to_global_phase(state: NDArray[np.complex128]) -> FloatArray:
    """Remove the global phase of a state whose amplitudes are real up to that phase."""
    k = int(np.argmax(np.abs(state)))
    mag = abs(state[k])
    phase = state[k] / mag if mag > 0 else 1.0
    real: FloatArray = (state * np.conj(phase)).real
    # The removed phase may have flipped the overall sign; fix it with the largest |+>-branch
    # amplitude, which is (c_i + c_{i+1}) / 2 >= 0 for non-negative pixel data.
    if real[0::2].sum() < 0:
        real = -real
    return real


def circuit_differences(
    flat_tiles: NDArray[np.floating], shots: int | None = None, seed: int | None = None
) -> FloatArray:
    """Neighbour differences for ``(B, N)`` tiles from QHED circuits run on Aer.

    Args:
        flat_tiles: Row-major flattened tiles with non-negative intensities, ``N`` a power
            of two.
        shots: ``None`` for exact statevector simulation (signed differences). An integer
            runs a shot-based simulation; probabilities only give magnitudes, so the result
            is ``|c_i - c_{i+1}|`` estimated from ``shots`` samples.
        seed: Simulator seed for reproducible shot sampling.

    Returns:
        Array of shape ``(B, N)`` matching :func:`q_edge.quantum.qhed_fast.qhed_differences`
        (up to sign and shot noise in shot mode). All-zero tiles give zeros.
    """
    if shots is not None and shots < 1:
        raise ValueError(f"shots must be a positive integer or None, got {shots}")
    amps, norms = normalize_tiles(flat_tiles)
    b, n_pix = amps.shape
    out = np.zeros((b, n_pix), dtype=np.float64)
    active = np.flatnonzero(norms > 0)
    if active.size == 0:
        return out

    circuits = []
    for idx in active:
        qc = build_qhed_circuit(amps[idx], measure=shots is not None)
        if shots is None:
            qc.save_statevector()
        circuits.append(qc)
    result = _run(circuits, shots, seed)

    for pos, idx in enumerate(active):
        if shots is None:
            state = np.asarray(result.get_statevector(pos), dtype=np.complex128)
            odd = _real_up_to_global_phase(state)[1::2]
        else:
            probs = np.zeros(2 * n_pix, dtype=np.float64)
            for key, count in result.get_counts(pos).items():
                probs[int(key.replace(" ", ""), 2)] = count / shots
            odd = np.sqrt(probs[1::2])
        out[idx] = odd * AMPLITUDE_TO_DIFFERENCE * norms[idx]
    return out


def circuit_responses(
    tiles: NDArray[np.floating], shots: int | None = None, seed: int | None = None
) -> tuple[FloatArray, FloatArray]:
    """Circuit-path horizontal and vertical responses for ``(B, T, T)`` tiles."""
    return oriented_responses(tiles, lambda flat: circuit_differences(flat, shots, seed))


def circuit_edge_magnitude(
    tiles: NDArray[np.floating], shots: int | None = None, seed: int | None = None
) -> FloatArray:
    """Circuit-path QHED edge magnitude, shape ``(B, T-1, T-1)``."""
    return combine(*circuit_responses(tiles, shots, seed))
