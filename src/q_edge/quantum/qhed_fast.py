"""Fast, exact, batched statevector simulation of QHED.

This path simulates exactly the same unitary as :mod:`q_edge.quantum.qhed_circuit`, but
on thousands of tiles at once with NumPy:

1. ``|psi> = |c> (x) |+>`` with the ancilla as the least-significant qubit.
2. The decrement permutation ``|k> -> |k - 1 mod 2N>`` over all ``n + 1`` qubits.
3. A Hadamard on the ancilla.

The odd-index amplitudes (ancilla ``|1>``) then equal ``(c_i - c_{i+1}) / 2``. Multiplying
by ``2 * norm`` recovers the true neighbour differences in the original intensity scale.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from q_edge.quantum.encoding import FloatArray, normalize_tiles

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

#: Factor that maps an ancilla-|1> amplitude back to a raw neighbour difference
#: (one 1/sqrt(2) from the initial |+> and one from the final Hadamard).
AMPLITUDE_TO_DIFFERENCE = 2.0

DifferenceFn = Callable[[FloatArray], FloatArray]


def qhed_statevector(amplitudes: NDArray[np.floating]) -> FloatArray:
    """Return the final ``(B, 2N)`` statevectors of the QHED circuit for a batch.

    Index ``k = 2 * i + a`` where ``i`` is the pixel index and ``a`` the ancilla bit.
    """
    amps = np.asarray(amplitudes, dtype=np.float64)
    psi = np.repeat(amps, 2, axis=1) * _INV_SQRT2  # |c> (x) |+>
    psi = np.roll(psi, -1, axis=1)  # decrement: new[k] = old[k + 1]
    even, odd = psi[:, 0::2], psi[:, 1::2]
    out: FloatArray = np.empty_like(psi)
    out[:, 0::2] = (even + odd) * _INV_SQRT2
    out[:, 1::2] = (even - odd) * _INV_SQRT2
    return out


def qhed_differences(flat_tiles: NDArray[np.floating]) -> FloatArray:
    """Return cyclic neighbour differences ``c_i - c_{(i+1) mod N}`` for ``(B, N)`` tiles.

    The values come from the simulated ancilla-|1> amplitudes rescaled by the tile norm.
    """
    amps, norms = normalize_tiles(flat_tiles)
    odd = qhed_statevector(amps)[:, 1::2]
    result: FloatArray = odd * (AMPLITUDE_TO_DIFFERENCE * norms[:, None])
    return result


def oriented_responses(
    tiles: NDArray[np.floating], differences: DifferenceFn
) -> tuple[FloatArray, FloatArray]:
    """Run a difference function horizontally and vertically on ``(B, T, T)`` tiles.

    Args:
        tiles: Square tiles, shape ``(B, T, T)``.
        differences: Function mapping ``(B, T*T)`` row-major tiles to cyclic neighbour
            differences of the same shape (fast or circuit implementation).

    Returns:
        ``(h, v)`` with ``h[b, r, c] = t[r, c] - t[r, c+1]`` of shape ``(B, T, T-1)`` and
        ``v[b, r, c] = t[r, c] - t[r+1, c]`` of shape ``(B, T-1, T)``. Terms that wrap
        across a row boundary are discarded.
    """
    batch = np.asarray(tiles, dtype=np.float64)
    if batch.ndim != 3 or batch.shape[1] != batch.shape[2]:
        raise ValueError(f"expected square tiles of shape (B, T, T), got {batch.shape}")
    b, t, _ = batch.shape
    h = differences(batch.reshape(b, t * t)).reshape(b, t, t)[:, :, : t - 1]
    transposed = np.ascontiguousarray(batch.transpose(0, 2, 1)).reshape(b, t * t)
    v = differences(transposed).reshape(b, t, t)[:, :, : t - 1].transpose(0, 2, 1)
    return h, v


def combine(h: FloatArray, v: FloatArray) -> FloatArray:
    """Combine oriented responses into an edge magnitude ``sqrt(h^2 + v^2)``.

    Returns the ``(B, T-1, T-1)`` core where both orientations are defined.
    """
    core = h.shape[1] - 1
    result: FloatArray = np.hypot(h[:, :core, :], v[:, :, :core])
    return result


def qhed_responses(tiles: NDArray[np.floating]) -> tuple[FloatArray, FloatArray]:
    """Fast-path horizontal and vertical QHED responses for ``(B, T, T)`` tiles."""
    return oriented_responses(tiles, qhed_differences)


def qhed_edge_magnitude(tiles: NDArray[np.floating]) -> FloatArray:
    """Fast-path QHED edge magnitude, shape ``(B, T-1, T-1)``."""
    return combine(*qhed_responses(tiles))
