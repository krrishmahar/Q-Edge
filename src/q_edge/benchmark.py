"""Side-by-side benchmarks of QHED against classical edge detectors.

These numbers come from *classical simulation* of the quantum algorithm. Simulation gives no
speedup over classical filters, and the benchmark reports that honestly. Its purpose is to
show that the quantum path is correct, that it scales linearly with image size through
tiling, and what the per-tile cost is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from q_edge.classical import CLASSICAL_METHODS, canny
from q_edge.config import Preset, QEdgeConfig
from q_edge.data.synthetic import synthetic_shapes
from q_edge.imaging import FloatArray
from q_edge.metrics.perf import measure, throughput
from q_edge.metrics.quality import edge_density, edge_scores, iou, psnr, ssim
from q_edge.pipeline import ImageSource, load_image, preprocess, run_pipeline

QUANTUM_METHOD = "qhed"
ALL_METHODS: tuple[str, ...] = (QUANTUM_METHOD, *CLASSICAL_METHODS)

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "method",
    "runtime_s",
    "tiles_per_s",
    "megapixels_per_s",
    "ssim",
    "psnr",
    "precision",
    "recall",
    "f1",
    "iou",
    "edge_density",
    "peak_mem_mb",
)

#: (height, width) sweep used for the runtime-vs-resolution chart.
DEFAULT_SCALING_SIZES: tuple[tuple[int, int], ...] = (
    (270, 480),
    (540, 960),
    (1080, 1920),
    (2160, 3840),
)


def detect_edges(
    image: FloatArray, method: str, config: QEdgeConfig
) -> tuple[FloatArray, NDArray[np.bool_], int]:
    """Run one method on a preprocessed image.

    Returns:
        ``(edge_map, binary, num_tiles)``; ``num_tiles`` is ``0`` for classical methods.
    """
    if method == QUANTUM_METHOD:
        result = run_pipeline(image, config.with_updates(preset=Preset.FULL_4K))
        return result.edges, result.binary, result.num_tiles
    try:
        detector = CLASSICAL_METHODS[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {ALL_METHODS}") from None
    edges = detector(image)
    return edges, edges >= max(config.threshold, 1e-12), 0


def benchmark(
    source: ImageSource,
    reference: NDArray[np.bool_] | None = None,
    methods: Sequence[str] = ALL_METHODS,
    config: QEdgeConfig | None = None,
    track_memory: bool = True,
) -> pd.DataFrame:
    """Compare QHED with classical detectors on one image.

    Args:
        source: Image to evaluate. It is preprocessed once with ``config.preset`` and every
            method sees the same pixels.
        reference: Ground-truth binary edge map at the preprocessed resolution. When
            ``None``, Canny's output is used as the reference (so Canny scores F1 = 1).
        methods: Methods to run; see :data:`ALL_METHODS`.
        config: Pipeline settings for the quantum method and the binarisation threshold.
        track_memory: Measure peak traced memory per method (adds some overhead).

    Returns:
        One row per method with :data:`BENCHMARK_COLUMNS`.
    """
    cfg = config or QEdgeConfig()
    image = preprocess(load_image(source), cfg.preset.max_side)
    ref = canny(image) > 0 if reference is None else np.asarray(reference, dtype=bool)
    if ref.shape != image.shape:
        raise ValueError(f"reference shape {ref.shape} does not match image {image.shape}")
    ref_float = ref.astype(np.float64)
    megapixels = image.size / 1e6

    rows = []
    for method in methods:
        with measure(track_memory) as perf:
            edges, binary, tiles = detect_edges(image, method, cfg)
        scores = edge_scores(binary, ref)
        rows.append(
            {
                "method": method,
                "runtime_s": perf.wall_time,
                "tiles_per_s": throughput(tiles, perf.wall_time) if tiles else np.nan,
                "megapixels_per_s": throughput(1, perf.wall_time) * megapixels,
                "ssim": ssim(edges, ref_float),
                "psnr": psnr(edges, ref_float),
                "precision": scores.precision,
                "recall": scores.recall,
                "f1": scores.f1,
                "iou": iou(binary, ref),
                "edge_density": edge_density(binary),
                "peak_mem_mb": perf.peak_memory_mb if track_memory else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(BENCHMARK_COLUMNS))


def scaling_benchmark(
    sizes: Sequence[tuple[int, int]] = DEFAULT_SCALING_SIZES,
    methods: Sequence[str] = (QUANTUM_METHOD, "sobel", "canny"),
    config: QEdgeConfig | None = None,
) -> pd.DataFrame:
    """Runtime versus resolution on synthetic scenes with ground-truth edges.

    Returns:
        A long-format DataFrame with ``height``, ``width``, ``megapixels`` and the
        :func:`benchmark` columns, one row per (size, method).
    """
    cfg = (config or QEdgeConfig()).with_updates(preset=Preset.FULL_4K)
    frames = []
    for height, width in sizes:
        image, truth = synthetic_shapes(height, width)
        df = benchmark(image, truth, methods, cfg, track_memory=False)
        df.insert(0, "megapixels", height * width / 1e6)
        df.insert(0, "width", width)
        df.insert(0, "height", height)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
