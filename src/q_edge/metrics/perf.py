"""Performance measurement helpers: wall time, throughput and peak memory."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class PerfSample:
    """Wall time (seconds) and peak traced memory (bytes) of a measured block."""

    wall_time: float = 0.0
    peak_memory: int = 0

    @property
    def peak_memory_mb(self) -> float:
        """Peak traced memory in MiB."""
        return self.peak_memory / 1024**2


@contextmanager
def measure(track_memory: bool = True) -> Iterator[PerfSample]:
    """Measure wall time and, optionally, peak Python/NumPy heap usage of a block.

    Memory is measured with :mod:`tracemalloc`, which sees NumPy allocations but not native
    allocations made by OpenCV or Aer, so treat it as a lower bound.
    """
    sample = PerfSample()
    started_tracing = track_memory and not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    if track_memory:
        tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        yield sample
    finally:
        sample.wall_time = time.perf_counter() - start
        if track_memory:
            sample.peak_memory = tracemalloc.get_traced_memory()[1]
        if started_tracing:
            tracemalloc.stop()


def throughput(count: int, seconds: float) -> float:
    """Items per second; ``inf`` when the elapsed time is zero."""
    return count / seconds if seconds > 0 else float("inf")
