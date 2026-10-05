"""Tests for the halo tiler and exact stitch-back."""

import numpy as np
import pytest

from q_edge.quantum.qhed_fast import qhed_edge_magnitude
from q_edge.tiling.tiler import split, stitch


def global_edge_magnitude(image: np.ndarray) -> np.ndarray:
    """Untiled reference: forward differences with edge replication."""
    padded = np.pad(image, ((0, 1), (0, 1)), mode="edge")
    h = padded[:-1, :-1] - padded[:-1, 1:]
    v = padded[:-1, :-1] - padded[1:, :-1]
    return np.hypot(h, v)


@pytest.mark.parametrize("tile_size", [4, 8, 16])
@pytest.mark.parametrize("shape", [(64, 64), (37, 53), (1, 1), (2, 9), (15, 16), (100, 7)])
def test_round_trip_is_lossless(tile_size: int, shape: tuple[int, int]) -> None:
    image = np.random.default_rng(0).random(shape)
    grid = split(image, tile_size)
    tiles = grid.all_tiles()
    assert tiles.shape == (grid.num_tiles, tile_size, tile_size)
    cores = tiles[:, : tile_size - 1, : tile_size - 1]
    np.testing.assert_array_equal(stitch(cores, grid), image)


@pytest.mark.parametrize("tile_size", [4, 8, 16])
@pytest.mark.parametrize("shape", [(64, 64), (37, 53), (2, 9), (129, 77)])
def test_tiled_qhed_matches_untiled_reference(tile_size: int, shape: tuple[int, int]) -> None:
    image = np.random.default_rng(1).random(shape)
    grid = split(image, tile_size)
    result = stitch(qhed_edge_magnitude(grid.all_tiles()), grid)
    assert result.shape == shape
    np.testing.assert_allclose(result, global_edge_magnitude(image), atol=1e-12)


def test_no_seams_on_smooth_ramp() -> None:
    # A linear ramp has a constant gradient everywhere; any seam would show up as a jump.
    w = 200
    image = np.tile(np.linspace(0.0, 1.0, w), (90, 1))
    grid = split(image, 8)
    result = stitch(qhed_edge_magnitude(grid.all_tiles()), grid)
    step = 1.0 / (w - 1)
    np.testing.assert_allclose(result[:, :-1], step, atol=1e-12)
    np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-12)


def test_grid_dimensions_and_padding() -> None:
    grid = split(np.zeros((37, 53)), 8)
    assert (grid.rows, grid.cols) == (6, 8)  # ceil(37/7), ceil(53/7)
    assert grid.core_size == 7
    assert grid.image_shape == (37, 53)


def test_batches_cover_all_tiles_in_order() -> None:
    image = np.random.default_rng(2).random((40, 33))
    grid = split(image, 4)
    ranges = list(grid.iter_batches(7))
    assert ranges[0][0] == 0 and ranges[-1][1] == grid.num_tiles
    stacked = np.concatenate([grid.batch(a, b) for a, b in ranges])
    np.testing.assert_array_equal(stacked, grid.all_tiles())


def test_tiler_input_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        split(np.zeros((4, 4, 3)), 4)
    with pytest.raises(ValueError, match="2-D"):
        split(np.zeros((0, 4)), 4)
    with pytest.raises(ValueError, match="tile_size"):
        split(np.zeros((4, 4)), 1)
    grid = split(np.zeros((8, 8)), 4)
    with pytest.raises(ValueError, match="cores"):
        stitch(np.zeros((1, 3, 3)), grid)
    with pytest.raises(ValueError, match="batch_size"):
        list(grid.iter_batches(0))
    with pytest.raises(ValueError, match="range"):
        grid.batch(5, 2)
