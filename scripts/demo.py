"""End-to-end Q-Edge demo.

Runs the bundled sample through the tiled QHED pipeline, cross-checks the fast NumPy path
against real Qiskit Aer circuits, processes a synthetic 4K frame, and prints benchmark
tables. Edge maps are written to ``outputs/``.

Usage: ``uv run python scripts/demo.py [--image PATH] [--scaling]``
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from q_edge.backends import AerBackend, NumpyBackend
from q_edge.benchmark import benchmark, scaling_benchmark
from q_edge.config import Preset, QEdgeConfig
from q_edge.data.synthetic import synthetic_shapes
from q_edge.pipeline import run_pipeline
from q_edge.ui import UploadError, decode_upload, to_png_bytes

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "app" / "assets" / "sample.png"
OUTPUT_DIR = ROOT / "outputs"

TABLE_COLUMNS = ["method", "runtime_s", "megapixels_per_s", "ssim", "psnr", "f1", "edge_density"]


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def show(df: pd.DataFrame) -> None:
    """Print a benchmark table with compact formatting."""
    print(df[TABLE_COLUMNS].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: list[str] | None = None) -> int:
    """Run the demo; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=SAMPLE, help="PNG/JPEG to process")
    parser.add_argument("--scaling", action="store_true", help="run the 480p-4K sweep")
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(exist_ok=True)

    section(f"1. Tiled QHED on {args.image.name}")
    try:
        decoded = decode_upload(args.image.read_bytes())
    except (OSError, UploadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = QEdgeConfig(tile_size=8, preset=Preset.BALANCED)
    result = run_pipeline(decoded.image, config)
    h, w = result.image.shape
    print(
        f"{w}x{h} px, {result.num_tiles:,} tiles of 8x8 "
        f"({int(math.log2(64)) + 1} qubits each), {result.total_time:.3f} s, "
        f"{result.tiles_per_second:,.0f} tiles/s, {result.workers} workers"
    )
    out = OUTPUT_DIR / f"{args.image.stem}_qhed_edges.png"
    out.write_bytes(to_png_bytes(result.edges))
    print(f"edge map written to {out.relative_to(ROOT)}")

    section("2. Fast path vs real Qiskit Aer circuits (64x64 crop, 4x4 tiles)")
    crop = result.image[:64, :64]
    small = QEdgeConfig(tile_size=4, workers=1)
    fast = run_pipeline(crop, small, backend=NumpyBackend()).magnitude
    start = time.perf_counter()
    circuit = run_pipeline(crop, small, backend=AerBackend()).magnitude
    elapsed = time.perf_counter() - start
    print(f"max |fast - circuit| = {np.abs(fast - circuit).max():.2e} (Aer took {elapsed:.2f} s)")

    section("3. Synthetic 4K frame (3840x2160) through the tiled pipeline")
    frame, _ = synthetic_shapes(2160, 3840)
    for tile in (8, 16):
        r = run_pipeline(frame, QEdgeConfig(tile_size=tile, preset=Preset.FULL_4K))
        print(
            f"tile {tile:>2}x{tile:<2}: {r.num_tiles:>7,} tiles, {r.total_time:.2f} s, "
            f"max circuit {int(math.log2(tile * tile)) + 1} qubits"
        )

    section(f"4. Benchmark on {args.image.name} (reference: Canny)")
    show(benchmark(result.image, config=config))

    section("5. Benchmark on synthetic 1080p (reference: ground truth)")
    image_1080, truth_1080 = synthetic_shapes(1080, 1920)
    show(benchmark(image_1080, truth_1080, config=config.with_updates(preset=Preset.FULL_4K)))

    if args.scaling:
        section("6. Runtime vs resolution")
        sweep = scaling_benchmark()
        print(
            sweep.pivot_table(index="megapixels", columns="method", values="runtime_s").to_string(
                float_format=lambda v: f"{v:.3f}"
            )
        )

    print(
        "\nNote: QHED runs on a classical simulator here. It is correct but slower than "
        "OpenCV; the prototype demonstrates architecture and scaling, not quantum advantage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
