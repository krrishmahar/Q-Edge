"""Tests for amplitude encoding and the two QHED implementations."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from q_edge.quantum.encoding import (
    amplitude_encoding_circuit,
    normalize_tiles,
    num_data_qubits,
)
from q_edge.quantum.qhed_circuit import (
    _real_up_to_global_phase,
    build_qhed_circuit,
    circuit_differences,
    circuit_edge_magnitude,
    circuit_responses,
    decrement_circuit,
)
from q_edge.quantum.qhed_fast import (
    qhed_differences,
    qhed_edge_magnitude,
    qhed_responses,
    qhed_statevector,
)


def _reference_responses(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = tiles[:, :, :-1] - tiles[:, :, 1:]
    v = tiles[:, :-1, :] - tiles[:, 1:, :]
    return h, v


# --- encoding ---------------------------------------------------------------------------


@pytest.mark.parametrize(("pixels", "qubits"), [(2, 1), (16, 4), (64, 6), (256, 8)])
def test_num_data_qubits(pixels: int, qubits: int) -> None:
    assert num_data_qubits(pixels) == qubits


@pytest.mark.parametrize("pixels", [0, 1, 3, 12, 100])
def test_num_data_qubits_rejects_non_power_of_two(pixels: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        num_data_qubits(pixels)


def test_normalize_tiles_unit_norm_and_row_major() -> None:
    tiles = np.arange(2 * 16, dtype=float).reshape(2, 4, 4)
    amps, norms = normalize_tiles(tiles)
    assert amps.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(amps, axis=1), 1.0)
    np.testing.assert_allclose(amps * norms[:, None], tiles.reshape(2, 16))


def test_normalize_zero_tile_has_no_nan() -> None:
    tiles = np.zeros((3, 4, 4))
    tiles[1] = 0.5
    amps, norms = normalize_tiles(tiles)
    assert np.all(np.isfinite(amps))
    assert norms[0] == 0.0 and norms[2] == 0.0
    np.testing.assert_array_equal(amps[0], 0.0)


def test_normalize_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="ndim"):
        normalize_tiles(np.ones(4))
    with pytest.raises(ValueError, match="power of two"):
        normalize_tiles(np.ones((1, 3, 3)))


def test_encoding_circuit_rejects_unnormalised() -> None:
    with pytest.raises(ValueError, match="unit L2 norm"):
        amplitude_encoding_circuit(np.zeros(4))


# --- decrement permutation --------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
def test_decrement_is_cyclic_shift(num_qubits: int) -> None:
    dim = 2**num_qubits
    unitary = Operator(decrement_circuit(num_qubits)).data
    expected = np.zeros((dim, dim))
    for k in range(dim):
        expected[(k - 1) % dim, k] = 1.0
    np.testing.assert_allclose(unitary, expected, atol=1e-12)


def test_decrement_rejects_zero_qubits() -> None:
    with pytest.raises(ValueError):
        decrement_circuit(0)


# --- fast path ----------------------------------------------------------------------------


def test_fast_statevector_is_normalised() -> None:
    amps, _ = normalize_tiles(np.random.default_rng(0).random((5, 16)))
    np.testing.assert_allclose(np.linalg.norm(qhed_statevector(amps), axis=1), 1.0)


def test_fast_gradient_tile_gives_expected_differences() -> None:
    # Horizontal ramp: each pixel is 0.1 brighter than its left neighbour.
    tile = np.tile(np.arange(8) * 0.1, (8, 1))[None]
    h, v = qhed_responses(tile)
    np.testing.assert_allclose(h, -0.1, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    np.testing.assert_allclose(qhed_edge_magnitude(tile), 0.1, atol=1e-12)


def test_fast_constant_tile_has_no_edges() -> None:
    tiles = np.full((4, 4, 4), 0.7)
    np.testing.assert_allclose(qhed_edge_magnitude(tiles), 0.0, atol=1e-12)


def test_fast_zero_tile() -> None:
    out = qhed_edge_magnitude(np.zeros((2, 8, 8)))
    assert out.shape == (2, 7, 7)
    assert np.all(np.isfinite(out)) and np.all(out == 0.0)


def test_fast_matches_classical_differences_on_random_tiles() -> None:
    tiles = np.random.default_rng(1).random((500, 16, 16))
    h, v = qhed_responses(tiles)
    h_ref, v_ref = _reference_responses(tiles)
    np.testing.assert_allclose(h, h_ref, atol=1e-12)
    np.testing.assert_allclose(v, v_ref, atol=1e-12)


def test_fast_cyclic_differences_include_wraparound() -> None:
    flat = np.array([[1.0, 2.0, 4.0, 8.0]])
    np.testing.assert_allclose(qhed_differences(flat), [[-1.0, -2.0, -4.0, 7.0]])


def test_fast_rejects_non_square_tiles() -> None:
    with pytest.raises(ValueError, match="square"):
        qhed_responses(np.ones((1, 4, 8)))


# --- circuit path -------------------------------------------------------------------------


@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_circuit_size_bounded_by_tile(tile_size: int) -> None:
    amps, _ = normalize_tiles(np.ones((1, tile_size, tile_size)))
    qc = build_qhed_circuit(amps[0])
    assert qc.num_qubits == int(np.log2(tile_size * tile_size)) + 1


@pytest.mark.parametrize("tile_size", [4, 8])
def test_fast_and_circuit_agree_on_random_tiles(tile_size: int) -> None:
    tiles = np.random.default_rng(tile_size).random((20, tile_size, tile_size))
    fast_h, fast_v = qhed_responses(tiles)
    circ_h, circ_v = circuit_responses(tiles)
    np.testing.assert_allclose(circ_h, fast_h, atol=1e-8)
    np.testing.assert_allclose(circ_v, fast_v, atol=1e-8)


def test_circuit_statevector_matches_fast_statevector() -> None:
    amps, _ = normalize_tiles(np.random.default_rng(7).random((1, 16)))
    state = Statevector(build_qhed_circuit(amps[0])).data
    np.testing.assert_allclose(
        _real_up_to_global_phase(state), qhed_statevector(amps)[0], atol=1e-8
    )


def test_circuit_zero_and_constant_tiles() -> None:
    tiles = np.zeros((2, 4, 4))
    tiles[1] = 0.3
    out = circuit_edge_magnitude(tiles)
    np.testing.assert_allclose(out, 0.0, atol=1e-8)


def test_circuit_shots_estimate_magnitudes() -> None:
    tiles = np.random.default_rng(3).random((3, 4, 4))
    exact = np.abs(qhed_differences(tiles.reshape(3, 16)))
    sampled = circuit_differences(tiles.reshape(3, 16), shots=200_000, seed=11)
    assert np.all(sampled >= 0)
    np.testing.assert_allclose(sampled, exact, atol=0.05)
    np.testing.assert_allclose(
        circuit_edge_magnitude(tiles, shots=200_000, seed=11),
        qhed_edge_magnitude(tiles),
        atol=0.05,
    )


def test_circuit_rejects_invalid_shots() -> None:
    with pytest.raises(ValueError, match="shots"):
        circuit_differences(np.ones((1, 4)), shots=0)
