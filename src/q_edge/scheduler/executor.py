"""Parallel execution of a backend over the tiles of a grid.

Tiles are independent, so the work is split into chunked batches (never one tile per task).
Only a bounded number of batches is in flight at a time, which keeps peak memory flat even
for 4K images. Results are written back by index, so output order never depends on
scheduling.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)

import numpy as np

from q_edge.backends.base import Backend, FloatArray
from q_edge.config import ExecutorKind
from q_edge.tiling.tiler import TileGrid

#: Receives ``(tiles_done, tiles_total)`` after every completed batch.
ProgressCallback = Callable[[int, int], None]

#: In-flight batches allowed per worker.
_IN_FLIGHT_PER_WORKER = 2


def _run_batch(backend: Backend, tiles: FloatArray) -> FloatArray:
    """Top-level (picklable) task body for worker processes."""
    return backend.run_tiles(tiles)


def execute(
    grid: TileGrid,
    backend: Backend,
    *,
    workers: int = 1,
    batch_size: int = 4096,
    kind: ExecutorKind = ExecutorKind.THREAD,
    progress: ProgressCallback | None = None,
) -> FloatArray:
    """Run ``backend`` over every tile in ``grid``.

    Args:
        grid: Tiles to process.
        backend: Backend computing ``(B, T-1, T-1)`` cores from ``(B, T, T)`` tiles.
        workers: Number of parallel workers; ``1`` (or ``kind=SERIAL``) runs in-process.
        batch_size: Tiles per task.
        kind: ``thread`` (default; NumPy and Aer release the GIL), ``process`` or ``serial``.
        progress: Optional callback with ``(done, total)`` tiles.

    Returns:
        Stacked cores of shape ``(num_tiles, T-1, T-1)`` in grid order.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    c = grid.core_size
    total = grid.num_tiles
    cores = np.empty((total, c, c), dtype=np.float64)
    done = 0

    def store(start: int, stop: int, result: FloatArray) -> None:
        nonlocal done
        cores[start:stop] = result
        done += stop - start
        if progress is not None:
            progress(done, total)

    ranges = list(grid.iter_batches(batch_size))
    if kind is ExecutorKind.SERIAL or workers == 1 or len(ranges) == 1:
        for start, stop in ranges:
            store(start, stop, backend.run_tiles(grid.batch(start, stop)))
        return cores

    pool: Executor
    if kind is ExecutorKind.PROCESS:
        pool = ProcessPoolExecutor(max_workers=workers)
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="q-edge")

    limit = workers * _IN_FLIGHT_PER_WORKER
    pending: dict[Future[FloatArray], tuple[int, int]] = {}
    with pool:
        for start, stop in ranges:
            if len(pending) >= limit:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    store(*pending.pop(fut), fut.result())
            fut = pool.submit(_run_batch, backend, grid.batch(start, stop))
            pending[fut] = (start, stop)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                store(*pending.pop(fut), fut.result())
    return cores
