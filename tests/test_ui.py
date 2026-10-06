"""Tests for UI helpers, chart builders and a headless run of the Streamlit app."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from q_edge.charts import (
    METHOD_COLORS,
    difference_heatmap,
    f1_by_method,
    runtime_vs_resolution,
)
from q_edge.config import QEdgeConfig
from q_edge.ui import (
    CIRCUIT_MAX_SIDE,
    MAX_UPLOAD_BYTES,
    UploadError,
    decode_upload,
    downscale,
    prepare_for_backend,
    preview,
    signed_difference,
    to_png_bytes,
)

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _encode(array: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt)
    return buf.getvalue()


# --- uploads ------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_decode_valid_upload(fmt: str) -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    decoded = decode_upload(_encode(rgb, fmt))
    assert decoded.format == fmt
    assert decoded.original_size == (60, 40)
    assert decoded.image.shape == (40, 60)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "empty"),
        (b"not an image at all", "not a readable"),
        (_encode(np.zeros((8, 8), dtype=np.uint8), "GIF"), "Unsupported format"),
        (_encode(np.zeros((1, 50), dtype=np.uint8)), "at least 2 pixels"),
        (_encode(np.zeros((10, 3841), dtype=np.uint8)), "4K"),
        (_encode(np.zeros((2161, 2161), dtype=np.uint8)), "4K"),
    ],
)
def test_decode_rejects_bad_uploads(data: bytes, message: str) -> None:
    with pytest.raises(UploadError, match=message):
        decode_upload(data)


def test_decode_rejects_truncated_file() -> None:
    data = _encode(np.random.default_rng(1).integers(0, 255, (64, 64), dtype=np.uint8))
    with pytest.raises(UploadError):
        decode_upload(data[: len(data) // 2])


def test_decode_rejects_oversized_file() -> None:
    with pytest.raises(UploadError, match="limit"):
        decode_upload(b"\0" * (MAX_UPLOAD_BYTES + 1))


# --- helpers ------------------------------------------------------------------------------


def test_prepare_for_backend_caps_circuit_backends() -> None:
    image = np.random.default_rng(0).random((300, 500))
    same, cfg, note = prepare_for_backend(image, QEdgeConfig())
    assert same is image and note is None and cfg.tile_size == 8
    small, cfg, note = prepare_for_backend(image, QEdgeConfig(backend="aer", tile_size=16))
    assert max(small.shape) == CIRCUIT_MAX_SIDE
    assert cfg.tile_size == 4 and note is not None and "aer" in note


def test_gpu_capabilities_is_json_safe() -> None:
    from q_edge.ui import gpu_capabilities

    capabilities = gpu_capabilities()
    assert {"available", "usable", "name", "reason"} <= capabilities.keys()
    assert isinstance(capabilities["usable"], bool)


def test_preview_and_downscale() -> None:
    big = np.random.default_rng(0).random((2000, 3000))
    out = preview(big, max_side=300)
    assert out.dtype == np.uint8 and out.shape == (200, 300)
    signed = downscale(-np.ones((40, 80)), 20)
    assert signed.shape == (10, 20) and np.allclose(signed, -1.0)
    small = np.ones((4, 4))
    assert downscale(small, 10) is not None and preview(small).shape == (4, 4)


def test_png_round_trip() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[2, 3] = True
    decoded = np.asarray(Image.open(io.BytesIO(to_png_bytes(mask))))
    assert decoded.shape == (5, 7) and decoded[2, 3] == 255 and decoded.sum() == 255


def test_signed_difference() -> None:
    np.testing.assert_allclose(signed_difference(np.ones((2, 2)), np.zeros((2, 2))), 1.0)
    with pytest.raises(ValueError, match="shape"):
        signed_difference(np.ones((2, 2)), np.ones((3, 3)))


# --- charts -------------------------------------------------------------------------------


def test_charts_build_with_fixed_colours() -> None:
    scaling = pd.DataFrame(
        {
            "height": [10, 20, 10, 20],
            "width": [10, 20, 10, 20],
            "megapixels": [1e-4, 4e-4, 1e-4, 4e-4],
            "method": ["qhed", "qhed", "sobel", "sobel"],
            "runtime_s": [0.1, 0.4, 0.01, 0.04],
        }
    )
    fig = runtime_vs_resolution(scaling)
    assert [t.line.color for t in fig.data] == [METHOD_COLORS["qhed"], METHOD_COLORS["sobel"]]
    bars = f1_by_method(pd.DataFrame({"method": ["qhed", "canny"], "f1": [1.0, 0.9]}))
    assert list(bars.data[0].marker.color) == [METHOD_COLORS["qhed"], METHOD_COLORS["canny"]]
    heat = difference_heatmap(np.zeros((3, 3)), "Sobel")
    assert heat.data[0].zmid == 0.0


# --- app ----------------------------------------------------------------------------------


def test_app_runs() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert not at.error



# --- hardening: unusual but valid uploads -------------------------------------------------


def test_decode_cmyk_jpeg() -> None:
    cmyk = Image.new("CMYK", (16, 12), (0, 0, 0, 0))
    buf = io.BytesIO()
    cmyk.save(buf, format="JPEG")
    decoded = decode_upload(buf.getvalue())
    assert decoded.image.shape == (12, 16)
    assert decoded.image.mean() > 0.9  # no ink = white


def test_decode_transparent_png_flattens_on_white() -> None:
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)  # fully transparent black
    rgba[:5, :, 3] = 255  # top half opaque black
    image = decode_upload(_encode(rgba)).image
    np.testing.assert_allclose(image[:5], 0.0)
    np.testing.assert_allclose(image[5:], 1.0)


def test_decode_16bit_png() -> None:
    arr = np.full((6, 6), 32768, dtype=np.uint16)
    image = decode_upload(_encode(arr)).image
    np.testing.assert_allclose(image, 32768 / 65535)


def test_decode_rejects_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(UploadError, match="too large"):
        decode_upload(_encode(np.zeros((20, 20), dtype=np.uint8)))


def test_baseline_benchmarks_structure() -> None:
    from q_edge.ui import get_baseline_benchmarks

    data = get_baseline_benchmarks()
    assert "sample_benchmarks" in data
    assert "synthetic_ground_truth" in data
    assert "scaling_sweep" in data
    assert "circuit_verification" in data
    assert len(data["sample_benchmarks"]) == 5
    assert len(data["scaling_sweep"]) == 4


def test_process_benchmark_request_synthetic() -> None:
    from q_edge.ui import process_benchmark_request

    payload = {
        "use_synthetic": True,
        "width": 64,
        "height": 64,
        "methods": ["qhed", "canny"],
    }
    resp = process_benchmark_request(payload)
    assert resp["status"] == "success"
    assert len(resp["results"]) == 2
    assert "summary" in resp
    assert resp["summary"]["total_methods"] == 2


def test_process_scaling_request_quick() -> None:
    from q_edge.ui import process_scaling_request

    payload = {"quick": True, "methods": ["qhed", "sobel"]}
    resp = process_scaling_request(payload)
    assert resp["status"] == "success"
    assert len(resp["pivot_rows"]) == 3
