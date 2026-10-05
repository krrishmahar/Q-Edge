"""Tests for configuration, image loading and the end-to-end pipeline."""

import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from q_edge.config import Preset, QEdgeConfig
from q_edge.data.synthetic import synthetic_shapes
from q_edge.imaging import normalise
from q_edge.pipeline import load_image, preprocess, run_pipeline


def _png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


# --- config -------------------------------------------------------------------------------


def test_config_defaults_and_presets() -> None:
    cfg = QEdgeConfig()
    assert cfg.tile_size == 8 and cfg.backend == "numpy"
    assert Preset.FAST.max_side == 512
    assert Preset.FULL_4K.max_side == 3840
    assert QEdgeConfig(preset="full-4k").preset is Preset.FULL_4K  # type: ignore[arg-type]
    assert cfg.with_updates(tile_size=16).tile_size == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": 5},
        {"workers": -1},
        {"batch_size": -2},
        {"shots": 0},
        {"threshold": 1.5},
        {"preset": "ultra"},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        QEdgeConfig(**kwargs)  # type: ignore[arg-type]


# --- loading and preprocessing ------------------------------------------------------------


def test_load_image_from_bytes_rgb_and_gray() -> None:
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    gray = load_image(_png_bytes(rgb))
    assert gray.shape == (10, 12)
    assert 0.25 < gray.mean() < 0.35  # red channel luma weight ~0.299


def test_load_image_from_path_and_pil(tmp_path: Path) -> None:
    arr = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 4).astype(np.uint8)
    path = tmp_path / "x.png"
    Image.fromarray(arr).save(path)
    np.testing.assert_allclose(load_image(path), arr / 255.0)
    np.testing.assert_allclose(load_image(Image.fromarray(arr)), arr / 255.0)


def test_load_image_16bit() -> None:
    arr = np.full((4, 4), 65535, dtype=np.uint16)
    np.testing.assert_allclose(load_image(_png_bytes(arr)), 1.0)


def test_load_image_from_arrays() -> None:
    np.testing.assert_allclose(load_image(np.full((3, 3), 255, dtype=np.uint8)), 1.0)
    rgba = np.ones((3, 3, 4))
    np.testing.assert_allclose(load_image(rgba), 1.0)
    with pytest.raises(ValueError, match="2-D"):
        load_image(np.ones((3, 3, 2)))


def test_preprocess_downscales_longest_side() -> None:
    image = np.random.default_rng(0).random((300, 600))
    out = preprocess(image, 200)
    assert out.shape == (100, 200)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert preprocess(image, 1000) is image


def test_normalise_handles_zero_and_noise_maps() -> None:
    np.testing.assert_array_equal(normalise(np.zeros((3, 3))), 0.0)
    np.testing.assert_array_equal(normalise(np.full((3, 3), 1e-15)), 0.0)
    np.testing.assert_allclose(normalise(np.array([[0.0, 2.0]])), [[0.0, 1.0]])


# --- pipeline -----------------------------------------------------------------------------


def test_pipeline_on_synthetic_shapes_finds_edges() -> None:
    image, truth = synthetic_shapes(96, 128)
    progress: list[tuple[int, int]] = []
    result = run_pipeline(
        image,
        QEdgeConfig(tile_size=8, batch_size=50),
        progress=lambda d, t: progress.append((d, t)),
    )
    assert result.edges.shape == image.shape
    assert result.edges.max() == pytest.approx(1.0)
    assert progress[-1] == (result.num_tiles, result.num_tiles)
    # Every detected edge pixel lies on a true region boundary.
    assert np.all(truth[result.binary])
    assert result.binary.sum() == truth.sum()
    assert result.tiles_per_second > 0
    assert set(result.timings) == {
        "load",
        "preprocess",
        "tile",
        "compute",
        "stitch",
        "postprocess",
    }


def test_pipeline_constant_image_has_no_edges() -> None:
    result = run_pipeline(np.full((20, 30), 0.5))
    assert not result.binary.any()
    assert np.all(result.edges == 0.0)


def test_pipeline_respects_preset() -> None:
    image = np.random.default_rng(0).random((1024, 2048))
    result = run_pipeline(image, QEdgeConfig(preset=Preset.FAST))
    assert result.image.shape == (256, 512)


@pytest.mark.slow
def test_pipeline_4k_end_to_end() -> None:
    image, truth = synthetic_shapes(2160, 3840)
    start = time.perf_counter()
    result = run_pipeline(image, QEdgeConfig(tile_size=8, preset=Preset.FULL_4K))
    elapsed = time.perf_counter() - start
    assert result.edges.shape == (2160, 3840)
    assert result.num_tiles == 309 * 549
    assert np.all(truth[result.binary])
    print(f"\n4K pipeline: {result.num_tiles} tiles in {elapsed:.2f}s")
