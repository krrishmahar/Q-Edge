"""Configuration objects for the Q-Edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from q_edge.quantum.encoding import SUPPORTED_TILE_SIZES


class Preset(StrEnum):
    """Resolution presets: images are downscaled so the longest side is at most this size."""

    FAST = "fast"
    BALANCED = "balanced"
    FULL_4K = "full-4k"

    @property
    def max_side(self) -> int:
        """Longest image side, in pixels, processed under this preset."""
        return _PRESET_MAX_SIDE[self]


_PRESET_MAX_SIDE: dict[Preset, int] = {
    Preset.FAST: 512,
    Preset.BALANCED: 1280,
    Preset.FULL_4K: 3840,
}


@dataclass(frozen=True)
class QEdgeConfig:
    """Immutable settings for one pipeline run.

    Attributes:
        tile_size: Tile side; one of :data:`SUPPORTED_TILE_SIZES`. A ``T x T`` tile uses
            ``log2(T*T) + 1`` qubits regardless of image size.
        backend: Name of a registered backend (``"numpy"``, ``"aer"``, ``"gpu"``, ...).
        workers: Parallel workers; ``0`` lets the scheduler choose.
        batch_size: Tiles per task; ``0`` lets the scheduler choose.
        shots: ``None`` for exact statevector simulation, otherwise shots per circuit
            (circuit backends only).
        threshold: Binarisation threshold in ``[0, 1]`` applied to the normalised edge map.
        preset: Resolution preset bounding the processed image size.
        seed: Optional random seed for shot-based simulation.
    """

    tile_size: int = 8
    backend: str = "numpy"
    workers: int = 0
    batch_size: int = 0
    shots: int | None = None
    threshold: float = 0.2
    preset: Preset = Preset.BALANCED
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.tile_size not in SUPPORTED_TILE_SIZES:
            raise ValueError(
                f"tile_size must be one of {SUPPORTED_TILE_SIZES}, got {self.tile_size}"
            )
        if self.workers < 0:
            raise ValueError(f"workers must be >= 0, got {self.workers}")
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"shots must be >= 1 or None, got {self.shots}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
        object.__setattr__(self, "preset", Preset(self.preset))

    def with_updates(self, **changes: object) -> QEdgeConfig:
        """Return a validated copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]
