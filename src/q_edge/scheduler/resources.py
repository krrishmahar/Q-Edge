"""Detect host resources and derive tile size, worker count and batch size."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import dataclass

from q_edge.backends.gpu_backend import gpu_status

_FALLBACK_MEMORY = 4 * 1024**3

#: Approximate bytes of working memory the fast path needs per tile pixel
#: (input copy, amplitudes, doubled statevector and temporaries, both orientations).
_BYTES_PER_PIXEL_FAST = 8 * 16

#: Fraction of available memory a single in-flight batch may use.
_BATCH_MEMORY_FRACTION = 0.02


@dataclass(frozen=True)
class SystemResources:
    """Snapshot of host resources relevant to scheduling."""

    cpu_count: int
    total_memory: int
    available_memory: int
    gpu_available: bool


@dataclass(frozen=True)
class AutoSettings:
    """Scheduler settings derived from the image size, backend and host resources."""

    tile_size: int
    workers: int
    batch_size: int


def _memory_windows() -> tuple[int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined,unused-ignore]
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _memory_posix() -> tuple[int, int]:
    if sys.platform == "win32":
        raise OSError("sysconf is not available on Windows")
    page = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES") * page
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * page
    except (ValueError, OSError):
        available = total // 2
    return int(total), int(available)


def detect_memory() -> tuple[int, int]:
    """Return ``(total, available)`` physical memory in bytes, with a safe fallback."""
    try:
        return _memory_windows() if sys.platform == "win32" else _memory_posix()
    except (OSError, ValueError, AttributeError):
        return _FALLBACK_MEMORY, _FALLBACK_MEMORY // 2


def detect_resources() -> SystemResources:
    """Detect CPU count, physical memory and CUDA GPU availability."""
    total, available = detect_memory()
    return SystemResources(
        cpu_count=os.cpu_count() or 1,
        total_memory=total,
        available_memory=available,
        gpu_available=gpu_status().usable,
    )


def auto_settings(
    image_shape: tuple[int, int],
    backend: str = "numpy",
    resources: SystemResources | None = None,
) -> AutoSettings:
    """Pick tile size, worker count and batch size for an image.

    Rules:
        * Circuit backends (``aer``, hardware) use 4x4 tiles (5 qubits) and small batches.
        * Simulation backends use 16x16 tiles for images above 2 MP (less halo overhead),
          8x8 otherwise.
        * The batch size keeps one in-flight batch to a small fraction of available memory.
        * Workers never exceed the CPU count or the number of batches.
    """
    res = resources or detect_resources()
    h, w = image_shape
    pixels = h * w
    circuit = backend.lower() in {"aer", "ibm-hardware"}
    gpu = backend.lower() == "gpu"

    if circuit:
        tile_size = 4
        batch_size = 32
    else:
        tile_size = 16 if pixels > 2_000_000 else 8
        budget = res.available_memory * _BATCH_MEMORY_FRACTION
        batch_size = int(budget // (tile_size * tile_size * _BYTES_PER_PIXEL_FAST))
        batch_size = max(256, min(batch_size, 65_536))
        if gpu:
            batch_size = max(batch_size, 4_096)

    stride = tile_size - 1
    num_tiles = math.ceil(h / stride) * math.ceil(w / stride)
    num_batches = math.ceil(num_tiles / batch_size)
    cpu_cap = min(res.cpu_count, 4) if circuit else res.cpu_count
    workers = 1 if gpu else max(1, min(cpu_cap, num_batches))
    return AutoSettings(tile_size=tile_size, workers=workers, batch_size=batch_size)
