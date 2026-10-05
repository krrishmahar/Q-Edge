"""Split images into overlapping tiles and stitch per-tile responses back together.

Scheme (resolution independent, seam free):

* Tiles are ``T x T`` and are taken with stride ``T - 1``, so neighbouring tiles share a
  1-pixel halo row/column.
* Each tile produces a ``(T - 1) x (T - 1)`` *core* response (the pixels for which both the
  right and the lower neighbour are inside the tile).
* Cores tile the image exactly without overlap, so stitching is a pure reshape.
* The image is edge-padded on the bottom/right so the grid covers it; the replicated border
  makes the last row/column behave like a global forward difference with edge replication.

Because every neighbour difference is computed inside some tile, the stitched result equals
the untiled computation exactly: there are no seam artifacts by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TileGrid:
    """A lazily-materialised grid of overlapping tiles over one image.

    Attributes:
        windows: Read-only strided view of shape ``(rows, cols, T, T)``.
        image_shape: ``(H, W)`` of the original, unpadded image.
        tile_size: Tile side ``T``.
    """

    windows: FloatArray
    image_shape: tuple[int, int]
    tile_size: int

    @property
    def rows(self) -> int:
        """Number of tile rows."""
        return int(self.windows.shape[0])

    @property
    def cols(self) -> int:
        """Number of tile columns."""
        return int(self.windows.shape[1])

    @property
    def num_tiles(self) -> int:
        """Total number of tiles."""
        return self.rows * self.cols

    @property
    def core_size(self) -> int:
        """Side of the per-tile core response, ``T - 1``."""
        return self.tile_size - 1

    def batch(self, start: int, stop: int) -> FloatArray:
        """Return a contiguous ``(stop - start, T, T)`` copy of tiles in row-major order."""
        stop = min(stop, self.num_tiles)
        if not 0 <= start <= stop:
            raise ValueError(f"invalid tile range [{start}, {stop})")
        rows, cols = np.divmod(np.arange(start, stop), self.cols)
        return np.ascontiguousarray(self.windows[rows, cols])

    def iter_batches(self, batch_size: int) -> Iterator[tuple[int, int]]:
        """Yield ``(start, stop)`` ranges that cover all tiles in chunks of ``batch_size``."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, self.num_tiles, batch_size):
            yield start, min(start + batch_size, self.num_tiles)

    def all_tiles(self) -> FloatArray:
        """Return every tile as one ``(num_tiles, T, T)`` array."""
        return self.batch(0, self.num_tiles)


def split(image: NDArray[np.floating], tile_size: int) -> TileGrid:
    """Split a 2-D image into overlapping ``tile_size`` tiles with a 1-pixel halo.

    Args:
        image: Grayscale image of shape ``(H, W)`` with ``H, W >= 1``.
        tile_size: Tile side ``T >= 2``.

    Returns:
        A :class:`TileGrid` whose cores cover the whole image.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"expected a non-empty 2-D image, got shape {img.shape}")
    if tile_size < 2:
        raise ValueError(f"tile_size must be >= 2, got {tile_size}")
    h, w = img.shape
    stride = tile_size - 1
    rows, cols = -(-h // stride), -(-w // stride)
    pad_h, pad_w = rows * stride + 1 - h, cols * stride + 1 - w
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    windows = sliding_window_view(padded, (tile_size, tile_size))[::stride, ::stride]
    return TileGrid(windows=windows, image_shape=(h, w), tile_size=tile_size)


def stitch(cores: NDArray[np.floating], grid: TileGrid) -> FloatArray:
    """Assemble per-tile core responses back into a full ``(H, W)`` map.

    Args:
        cores: Array of shape ``(num_tiles, T-1, T-1)`` in the grid's row-major tile order.
        grid: The grid the cores were computed from.
    """
    c = grid.core_size
    arr = np.asarray(cores, dtype=np.float64)
    if arr.shape != (grid.num_tiles, c, c):
        raise ValueError(f"expected cores of shape {(grid.num_tiles, c, c)}, got {arr.shape}")
    full = arr.reshape(grid.rows, grid.cols, c, c).transpose(0, 2, 1, 3)
    h, w = grid.image_shape
    result: FloatArray = full.reshape(grid.rows * c, grid.cols * c)[:h, :w]
    return np.ascontiguousarray(result)
