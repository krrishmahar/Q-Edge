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


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    """Return qubit count, circuit depth, and total gate count for a circuit.

    Args:
        circuit: A Qiskit QuantumCircuit.
    """
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
    }


def build_noise_model(p_depol: float = 0.01, p_readout: float = 0.02) -> Any:
    """Build a configurable depolarizing and readout noise model for Aer simulation.

    Args:
        p_depol: 1-qubit and 2-qubit depolarizing error probability.
        p_readout: Readout measurement bit-flip error probability.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    if p_depol > 0:
        err_1q = depolarizing_error(p_depol, 1)
        err_2q = depolarizing_error(min(p_depol * 2, 1.0), 2)
        nm.add_all_qubit_quantum_error(err_1q, ["h", "x"])
        nm.add_all_qubit_quantum_error(err_2q, ["cx", "mcx"])
    if p_readout > 0:
        ro = ReadoutError([[1.0 - p_readout, p_readout], [p_readout, 1.0 - p_readout]])
        nm.add_all_qubit_readout_error(ro)
    return nm


def _run(
    circuits: Sequence[QuantumCircuit],
    shots: int | None,
    seed: int | None,
    noise_model: Any = None,
) -> Any:
    """Transpile and run circuits on a statevector or noisy Aer simulator."""
    sim = AerSimulator(
        method="statevector" if noise_model is None else "automatic",
        noise_model=noise_model,
        seed_simulator=seed,
    )
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
    flat_tiles: NDArray[np.floating],
    shots: int | None = None,
    seed: int | None = None,
    noise_model: Any = None,
) -> FloatArray:
    """Neighbour differences for ``(B, N)`` tiles from QHED circuits run on Aer.

    Args:
        flat_tiles: Row-major flattened tiles with non-negative intensities, ``N`` a power
            of two.
        shots: ``None`` for exact statevector simulation (signed differences). An integer
            runs a shot-based simulation; probabilities only give magnitudes, so the result
            is ``|c_i - c_{i+1}|`` estimated from ``shots`` samples.
        seed: Simulator seed for reproducible shot sampling.
        noise_model: Optional Qiskit Aer NoiseModel for noisy simulation.

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

    # If noise model is supplied, shot-based sampling with measurement is required
    effective_shots = shots or (4096 if noise_model is not None else None)
    circuits = []
    for idx in active:
        qc = build_qhed_circuit(amps[idx], measure=effective_shots is not None)
        if effective_shots is None:
            qc.save_statevector()
        circuits.append(qc)
    result = _run(circuits, effective_shots, seed, noise_model=noise_model)

    for pos, idx in enumerate(active):
        if effective_shots is None:
            state = np.asarray(result.get_statevector(pos), dtype=np.complex128)
            odd = _real_up_to_global_phase(state)[1::2]
        else:
            probs = np.zeros(2 * n_pix, dtype=np.float64)
            for key, count in result.get_counts(pos).items():
                probs[int(key.replace(" ", ""), 2)] = count / effective_shots
            odd = np.sqrt(probs[1::2])
        out[idx] = odd * AMPLITUDE_TO_DIFFERENCE * norms[idx]
    return out


def circuit_responses(
    tiles: NDArray[np.floating],
    shots: int | None = None,
    seed: int | None = None,
    noise_model: Any = None,
) -> tuple[FloatArray, FloatArray]:
    """Circuit-path horizontal and vertical responses for ``(B, T, T)`` tiles."""
    return oriented_responses(
        tiles,
        lambda flat: circuit_differences(flat, shots, seed, noise_model=noise_model),
    )


def circuit_edge_magnitude(
    tiles: NDArray[np.floating],
    shots: int | None = None,
    seed: int | None = None,
    noise_model: Any = None,
) -> FloatArray:
    """Circuit-path QHED edge magnitude, shape ``(B, T-1, T-1)``."""
    return combine(*circuit_responses(tiles, shots, seed, noise_model=noise_model))
