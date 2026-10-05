"""End-to-end pipeline: load, preprocess, tile, run backend, stitch, post-process."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from q_edge.backends import Backend, get_backend
from q_edge.config import QEdgeConfig
from q_edge.scheduler.executor import ProgressCallback, execute
from q_edge.scheduler.resources import auto_settings
from q_edge.tiling.tiler import split, stitch

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

ImageSource = str | Path | bytes | Image.Image | NDArray[np.generic]


@dataclass(frozen=True)
class PipelineResult:
    """Output of one pipeline run.

    Attributes:
        image: Preprocessed grayscale input in ``[0, 1]``, shape ``(H, W)``.
        magnitude: Raw edge magnitude ``sqrt(h^2 + v^2)`` in intensity units.
        edges: Edge map normalised to ``[0, 1]``.
        binary: ``edges >= threshold``.
        num_tiles: Number of tiles processed.
        workers: Workers actually used.
        batch_size: Tiles per batch actually used.
        timings: Wall time in seconds per stage.
        config: The configuration used.
    """

    image: FloatArray
    magnitude: FloatArray
    edges: FloatArray
    binary: BoolArray
    num_tiles: int
    workers: int = 1
    batch_size: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    config: QEdgeConfig = field(default_factory=QEdgeConfig)

    @property
    def total_time(self) -> float:
        """Total wall time over all stages, in seconds."""
        return sum(self.timings.values())

    @property
    def tiles_per_second(self) -> float:
        """Throughput of the tile-processing stage."""
        compute = self.timings.get("compute", 0.0)
        return self.num_tiles / compute if compute > 0 else float("inf")


def load_image(source: ImageSource) -> FloatArray:
    """Load an image as a float64 grayscale array in ``[0, 1]``.

    Accepts a file path, raw encoded bytes, a PIL image, or a NumPy array (2-D grayscale,
    or 3-D RGB/RGBA; integer arrays are scaled by their dtype's maximum).
    """
    if isinstance(source, np.ndarray):
        return _array_to_gray(source)
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float64)
        return np.clip(arr / 65535.0, 0.0, 1.0)
    return np.asarray(img.convert("L"), dtype=np.float64) / 255.0


def _array_to_gray(arr: NDArray[np.generic]) -> FloatArray:
    data = np.asarray(arr)
    if np.issubdtype(data.dtype, np.integer):
        scaled = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        scaled = data.astype(np.float64)
    if scaled.ndim == 3 and scaled.shape[2] in (3, 4):
        scaled = scaled[..., :3] @ np.array([0.299, 0.587, 0.114])
    if scaled.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale or RGB(A) image, got shape {data.shape}")
    return np.clip(scaled, 0.0, 1.0)


def preprocess(image: FloatArray, max_side: int) -> FloatArray:
    """Downscale (area interpolation) so the longest side is at most ``max_side``."""
    h, w = image.shape
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return np.clip(resized.astype(np.float64), 0.0, 1.0)


def normalise(magnitude: FloatArray) -> FloatArray:
    """Scale an edge magnitude map to ``[0, 1]`` by its maximum (all-zero maps stay zero)."""
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    return magnitude / peak if peak > 0 else np.zeros_like(magnitude)


def run_pipeline(
    source: ImageSource,
    config: QEdgeConfig | None = None,
    backend: Backend | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Run QHED edge detection on an image through the tiled pipeline.

    Args:
        source: Image to process (see :func:`load_image`).
        config: Pipeline settings; defaults to :class:`QEdgeConfig`.
        backend: Backend instance; defaults to the registered backend named by
            ``config.backend``.
        progress: Optional callback invoked with ``(done, total)`` tiles.

    Returns:
        A :class:`PipelineResult` with the edge map, binary mask and stage timings.
    """
    cfg = config or QEdgeConfig()
    engine = backend or get_backend(cfg.backend, cfg)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    image = load_image(source)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = preprocess(image, cfg.preset.max_side)
    timings["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = split(image, cfg.tile_size)
    timings["tile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    auto = auto_settings(image.shape, engine.name)
    workers = cfg.workers or auto.workers
    batch_size = cfg.batch_size or auto.batch_size
    cores = execute(
        grid,
        engine,
        workers=workers,
        batch_size=batch_size,
        kind=cfg.executor,
        progress=progress,
    )
    timings["compute"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    magnitude = stitch(cores, grid)
    timings["stitch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = normalise(magnitude)
    binary = edges >= cfg.threshold if cfg.threshold > 0 else edges > 0
    timings["postprocess"] = time.perf_counter() - t0

    return PipelineResult(
        image=image,
        magnitude=magnitude,
        edges=edges,
        binary=binary,
        num_tiles=grid.num_tiles,
        workers=workers,
        batch_size=batch_size,
        timings=timings,
        config=cfg,
    )
