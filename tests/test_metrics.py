"""Tests for classical detectors, quality/performance metrics and the benchmark."""

import math
import time

import numpy as np
import pytest

from q_edge.benchmark import (
    ALL_METHODS,
    BENCHMARK_COLUMNS,
    benchmark,
    detect_edges,
    scaling_benchmark,
)
from q_edge.classical import CLASSICAL_METHODS, canny
from q_edge.classical.canny import auto_thresholds
from q_edge.config import QEdgeConfig
from q_edge.data.synthetic import label_edges, synthetic_shapes
from q_edge.metrics import edge_density, edge_scores, measure, psnr, ssim, throughput

IMAGE, TRUTH = synthetic_shapes(96, 128)


# --- classical detectors ------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CLASSICAL_METHODS))
def test_classical_detectors_output_range(name: str) -> None:
    edges = CLASSICAL_METHODS[name](IMAGE)
    assert edges.shape == IMAGE.shape
    assert edges.min() >= 0.0 and edges.max() == pytest.approx(1.0)


@pytest.mark.parametrize("name", list(CLASSICAL_METHODS))
def test_classical_detectors_constant_image(name: str) -> None:
    edges = CLASSICAL_METHODS[name](np.full((32, 32), 0.4))
    assert np.all(edges == 0.0)


def test_classical_detectors_find_true_edges() -> None:
    for name, detector in CLASSICAL_METHODS.items():
        scores = edge_scores(detector(IMAGE) >= 0.2, TRUTH, tolerance=2)
        assert scores.recall > 0.9, name


def test_canny_thresholds() -> None:
    low, high = auto_thresholds(np.zeros((4, 4), dtype=np.uint8))
    assert low == 10.0 and high == 20.0
    low, high = auto_thresholds(np.full((4, 4), 200, dtype=np.uint8))
    assert low < high <= 255.0
    assert canny(IMAGE, low=5, high=50, blur=0).any()


# --- quality metrics ----------------------------------------------------------------------


def test_identical_maps_score_perfectly() -> None:
    edges = TRUTH.astype(float)
    assert ssim(edges, edges) == pytest.approx(1.0)
    assert math.isinf(psnr(edges, edges))
    assert edge_scores(TRUTH, TRUTH) == edge_scores(TRUTH, TRUTH, tolerance=0)
    assert edge_scores(TRUTH, TRUTH).f1 == 1.0


def test_psnr_known_value() -> None:
    a = np.zeros((10, 10))
    b = np.full((10, 10), 0.1)
    assert psnr(a, b) == pytest.approx(20.0)


def test_ssim_tiny_images() -> None:
    assert ssim(np.ones((2, 2)), np.ones((2, 2))) == 1.0
    assert ssim(np.ones((2, 2)), np.zeros((2, 2))) == 0.0
    assert -1.0 <= ssim(np.random.default_rng(0).random((5, 5)), np.zeros((5, 5))) <= 1.0


def test_edge_scores_pixel_tolerance() -> None:
    ref = np.zeros((20, 20), dtype=bool)
    ref[:, 10] = True
    shifted = np.roll(ref, 1, axis=1)
    assert edge_scores(shifted, ref, tolerance=0).f1 == 0.0
    assert edge_scores(shifted, ref, tolerance=1).f1 == 1.0


def test_edge_scores_empty_cases() -> None:
    empty = np.zeros((5, 5), dtype=bool)
    full = np.ones((5, 5), dtype=bool)
    assert edge_scores(empty, empty).f1 == 1.0
    assert edge_scores(empty, full).f1 == 0.0
    assert edge_scores(full, empty).precision == 0.0


def test_metric_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        edge_scores(np.zeros((2, 2), bool), np.zeros((3, 3), bool))
    with pytest.raises(ValueError, match="shape"):
        psnr(np.zeros(2), np.zeros(3))


def test_edge_density() -> None:
    assert edge_density(np.array([[True, False], [False, False]])) == 0.25
    assert edge_density(np.zeros((0,), dtype=bool)) == 0.0


def test_label_edges_marks_boundaries() -> None:
    labels = np.zeros((4, 4), dtype=int)
    labels[:, 2:] = 1
    np.testing.assert_array_equal(label_edges(labels)[:, 1], True)
    assert label_edges(labels).sum() == 4


# --- performance metrics ------------------------------------------------------------------


def test_measure_records_time_and_memory() -> None:
    with measure() as sample:
        block = np.ones((512, 512))
        time.sleep(0.01)
    assert sample.wall_time >= 0.01
    assert sample.peak_memory >= block.nbytes
    assert sample.peak_memory_mb > 0
    with measure(track_memory=False) as untracked:
        pass
    assert untracked.peak_memory == 0


def test_throughput() -> None:
    assert throughput(10, 2.0) == 5.0
    assert math.isinf(throughput(1, 0.0))


# --- benchmark ----------------------------------------------------------------------------


def test_benchmark_dataframe_with_ground_truth() -> None:
    df = benchmark(IMAGE, TRUTH, config=QEdgeConfig(tile_size=8))
    assert list(df.columns) == list(BENCHMARK_COLUMNS)
    assert list(df["method"]) == list(ALL_METHODS)
    qhed = df.set_index("method").loc["qhed"]
    assert qhed["f1"] == pytest.approx(1.0)
    assert qhed["tiles_per_s"] > 0
    assert df["runtime_s"].gt(0).all()
    assert df.loc[df["method"] != "qhed", "tiles_per_s"].isna().all()


def test_benchmark_defaults_to_canny_reference() -> None:
    df = benchmark(IMAGE, methods=("qhed", "canny"), track_memory=False)
    assert df.set_index("method").loc["canny", "f1"] == pytest.approx(1.0)
    assert df["peak_mem_mb"].isna().all()


def test_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="reference shape"):
        benchmark(IMAGE, np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="unknown method"):
        detect_edges(IMAGE, "magic", QEdgeConfig())


def test_scaling_benchmark_long_format() -> None:
    df = scaling_benchmark(sizes=((64, 64), (128, 256)), methods=("qhed", "sobel"))
    assert len(df) == 4
    assert {"height", "width", "megapixels", "method", "runtime_s"} <= set(df.columns)
    assert df.loc[df["method"] == "qhed", "f1"].eq(1.0).all()
