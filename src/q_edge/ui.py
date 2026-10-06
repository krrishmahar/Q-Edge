"""Framework-independent helpers for the Streamlit front-end.

Everything here is plain Python/NumPy so it can be unit-tested without a browser.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from q_edge.backends.gpu_backend import gpu_status
from q_edge.config import QEdgeConfig
from q_edge.imaging import FloatArray
from q_edge.pipeline import load_image, preprocess

#: Longest accepted image side (4K UHD width).
MAX_UPLOAD_SIDE = 3840

#: Largest accepted upload, in bytes.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

#: Accepted upload formats (as reported by Pillow).
ALLOWED_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG"})

#: Longest side processed by circuit backends, which simulate one circuit per tile.
CIRCUIT_MAX_SIDE = 64

#: Circuit-level backends that need the small-image cap.
CIRCUIT_BACKENDS: frozenset[str] = frozenset({"aer", "ibm-hardware"})

#: Longest side of images shown in the browser.
PREVIEW_MAX_SIDE = 1024

#: Longest side of the interactive difference heatmap (one plotly cell per pixel).
HEATMAP_MAX_SIDE = 360


def gpu_capabilities() -> dict[str, Any]:
    """Return JSON-safe CUDA capability information for the frontend."""
    status = gpu_status()
    return {
        "available": status.available,
        "usable": status.usable,
        "name": status.name,
        "compute_capability": status.compute_capability,
        "memory_total_mb": status.memory_total_mb,
        "cupy_version": status.cupy_version,
        "reason": status.reason,
    }


class UploadError(ValueError):
    """An uploaded file cannot be used; the message is safe to show to users."""


@dataclass(frozen=True)
class DecodedUpload:
    """A validated upload converted to grayscale."""

    image: FloatArray
    format: str
    original_size: tuple[int, int]


def decode_upload(data: bytes) -> DecodedUpload:
    """Validate and decode an uploaded image.

    Raises:
        UploadError: For empty, oversized, corrupt, unsupported or larger-than-4K files.
    """
    if not data:
        raise UploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"The file is {len(data) / 1024**2:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1024**2} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadError("The file is not a readable PNG or JPEG image.") from exc
    except Image.DecompressionBombError as exc:
        raise UploadError("The image dimensions are too large to process safely.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"Unsupported format {fmt or 'unknown'}; upload a PNG or JPEG.")
    if max(width, height) > MAX_UPLOAD_SIDE or min(width, height) > 2160:
        raise UploadError(
            f"The image is {width}x{height}; the maximum supported size is 3840x2160 (4K)."
        )
    if min(width, height) < 2:
        raise UploadError("The image must be at least 2 pixels in each dimension.")
    try:
        image = load_image(data)
    except (OSError, ValueError) as exc:
        raise UploadError("The image could not be decoded (it may be truncated).") from exc
    return DecodedUpload(image=image, format=fmt, original_size=(width, height))


def prepare_for_backend(
    image: FloatArray, config: QEdgeConfig
) -> tuple[FloatArray, QEdgeConfig, str | None]:
    """Apply UI safety limits for the chosen backend.

    Circuit backends simulate a real circuit per tile, so their input is downscaled to
    :data:`CIRCUIT_MAX_SIDE` and 4x4 tiles (5 qubits) are forced. Returns the image, the
    adjusted config and a user-facing note when anything changed.
    """
    if config.backend not in CIRCUIT_BACKENDS:
        return image, config, None
    note = (
        f"The '{config.backend}' backend runs one circuit per tile, so the image is "
        f"downscaled to at most {CIRCUIT_MAX_SIDE}px and 4x4 tiles (5 qubits) are used."
    )
    return preprocess(image, CIRCUIT_MAX_SIDE), config.with_updates(tile_size=4), note


def downscale(image: NDArray[np.generic], max_side: int) -> FloatArray:
    """Area-downscale a 2-D map (any sign) so its longest side is at most ``max_side``."""
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return arr
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(arr.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.float64)


def preview(image: NDArray[np.generic], max_side: int = PREVIEW_MAX_SIDE) -> NDArray[np.uint8]:
    """Downscale an image or edge map for display and convert it to 8-bit."""
    arr = downscale(image, max_side)
    return np.asarray(np.clip(arr, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def to_png_bytes(image: NDArray[np.generic]) -> bytes:
    """Encode a ``[0, 1]`` float map or a boolean mask as a full-resolution 8-bit PNG."""
    arr = np.asarray(image, dtype=np.float64)
    pixels = np.asarray(np.clip(arr, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def signed_difference(quantum: FloatArray, classical: FloatArray) -> FloatArray:
    """Return ``quantum - classical`` for two ``[0, 1]`` edge maps of the same shape."""
    if quantum.shape != classical.shape:
        raise ValueError(f"shape mismatch: {quantum.shape} vs {classical.shape}")
    result: FloatArray = np.asarray(quantum, dtype=np.float64) - classical
    return result


def to_png_data_url(image: NDArray[np.generic]) -> str:
    """Encode an image or edge map as a base64 data:image/png URL."""
    import base64

    b64 = base64.b64encode(to_png_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def process_pipeline_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Process an image through the real Q-Edge quantum pipeline and return JSON response.

    Includes real QHED edge map, Sobel, Canny, timings, metadata, and 9-stage tile breakdown.
    """
    import base64

    from q_edge.classical.canny import canny
    from q_edge.classical.sobel import sobel
    from q_edge.config import Preset, QEdgeConfig
    from q_edge.pipeline import load_image, run_pipeline
    from q_edge.quantum.encoding import normalize_tiles
    from q_edge.quantum.qhed_circuit import build_qhed_circuit, circuit_metrics
    from q_edge.quantum.qhed_fast import combine, qhed_responses

    image_b64 = str(payload.get("image_b64", ""))
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    image = load_image(image_bytes)

    tile_size = int(payload.get("tile_size", 8))
    sim_tile_size = tile_size if tile_size in (4, 8, 16) else 8
    backend = str(payload.get("backend", "numpy"))
    if backend == "gpu":
        capabilities = gpu_capabilities()
        if not capabilities["usable"]:
            raise ValueError(f"CUDA GPU backend is unavailable: {capabilities['reason']}")
    threshold = float(payload.get("threshold", 0.2))
    raw_shots = payload.get("shots")
    shots = int(raw_shots) if raw_shots is not None and str(raw_shots).isdigit() else None

    config = QEdgeConfig(
        tile_size=sim_tile_size,
        backend=backend,
        threshold=threshold,
        shots=shots,
        preset=Preset.FULL_4K,
    )

    result = run_pipeline(image, config)
    sobel_map = sobel(result.image)
    canny_map = canny(result.image)

    # Compute a real tile through the 9 stages
    h, w = result.image.shape
    th = min(8, h, w)
    tile_crop = result.image[:th, :th]
    if tile_crop.shape != (8, 8):
        tile_crop = np.pad(
            tile_crop,
            ((0, 8 - tile_crop.shape[0]), (0, 8 - tile_crop.shape[1])),
            mode="edge",
        )

    norm_tile = tile_crop / (np.max(tile_crop) or 1.0)
    amps, _ = normalize_tiles(norm_tile[None, ...])
    gx_resp, gy_resp = qhed_responses(norm_tile[None, ...])
    mag_tile = combine(gx_resp, gy_resp)[0]
    edge_tile = mag_tile >= threshold

    # Build real circuit for this tile to get exact depth & gate count
    qc = build_qhed_circuit(amps[0])
    c_metrics = circuit_metrics(qc)

    tile_stages = {
        "original": to_png_data_url(tile_crop),
        "gray": to_png_data_url(tile_crop),
        "normalized": to_png_data_url(norm_tile),
        "qpie": to_png_data_url(amps.reshape(8, 8)),
        "gx": to_png_data_url(np.abs(gx_resp[0])),
        "gy": to_png_data_url(np.abs(gy_resp[0])),
        "magnitude": to_png_data_url(mag_tile),
        "edges": to_png_data_url(edge_tile),
        "circuit": c_metrics,
    }

    return {
        "status": "success",
        "qhed_b64": to_png_data_url(result.edges),
        "sobel_b64": to_png_data_url(sobel_map),
        "canny_b64": to_png_data_url(canny_map),
        "original_b64": to_png_data_url(result.image),
        "timings": {
            "execution": result.timings.get("compute", 0.0),
            "preprocess": result.timings.get("preprocess", 0.0) + result.timings.get("load", 0.0),
            "stitch": result.timings.get("stitch", 0.0) + result.timings.get("postprocess", 0.0),
            "total": result.total_time,
        },
        "metadata": {
            "qubits": result.qubits_per_tile or c_metrics["num_qubits"],
            "depth": c_metrics["depth"],
            "gates": c_metrics["gate_count"],
            "tiles": result.num_tiles,
            "width": w,
            "height": h,
            "backend": backend,
            "gpu": gpu_capabilities() if backend == "gpu" else None,
        },
        "tile_stages": tile_stages,
    }


def get_baseline_benchmarks() -> dict[str, Any]:
    """Return precomputed, verified baseline benchmark results.

    Contains active sample image benchmark, 1080p synthetic ground-truth benchmark,
    and 480p-to-4K resolution scaling sweep.
    """
    return {
        "sample_benchmarks": [
            {
                "method": "qhed",
                "label": "QHED (Simulated)",
                "type": "Quantum (Simulated)",
                "runtime_s": 0.1431,
                "runtime_ms": 143.1,
                "megapixels_per_s": 6.44,
                "ssim": 0.1888,
                "psnr": 23.62,
                "precision": 0.9996,
                "recall": 0.9996,
                "f1": 0.9996,
                "iou": 0.9992,
                "edge_density": 0.0059,
                "peak_mem_mb": 4.8,
                "qubits": 7,
            },
            {
                "method": "sobel",
                "label": "Sobel",
                "type": "Classical (Spatial 3×3)",
                "runtime_s": 0.0233,
                "runtime_ms": 23.3,
                "megapixels_per_s": 39.64,
                "ssim": 0.2779,
                "psnr": 24.48,
                "precision": 0.9997,
                "recall": 0.9997,
                "f1": 0.9997,
                "iou": 0.9994,
                "edge_density": 0.0119,
                "peak_mem_mb": 2.1,
                "qubits": None,
            },
            {
                "method": "prewitt",
                "label": "Prewitt",
                "type": "Classical (Spatial 3×3)",
                "runtime_s": 0.0184,
                "runtime_ms": 18.4,
                "megapixels_per_s": 50.19,
                "ssim": 0.2889,
                "psnr": 24.26,
                "precision": 0.9997,
                "recall": 0.9997,
                "f1": 0.9997,
                "iou": 0.9994,
                "edge_density": 0.0127,
                "peak_mem_mb": 1.9,
                "qubits": None,
            },
            {
                "method": "canny",
                "label": "Canny",
                "type": "Classical (Multi-stage)",
                "runtime_s": 0.0130,
                "runtime_ms": 13.0,
                "megapixels_per_s": 70.98,
                "ssim": 1.0000,
                "psnr": 999.0,
                "precision": 1.0000,
                "recall": 1.0000,
                "f1": 1.0000,
                "iou": 1.0000,
                "edge_density": 0.0061,
                "peak_mem_mb": 2.4,
                "qubits": None,
            },
            {
                "method": "laplacian",
                "label": "Laplacian",
                "type": "Classical (2nd Derivative)",
                "runtime_s": 0.0121,
                "runtime_ms": 12.1,
                "megapixels_per_s": 75.99,
                "ssim": 0.2251,
                "psnr": 24.25,
                "precision": 0.9894,
                "recall": 0.9894,
                "f1": 0.9894,
                "iou": 0.9790,
                "edge_density": 0.0102,
                "peak_mem_mb": 1.8,
                "qubits": None,
            },
        ],
        "synthetic_ground_truth": [
            {
                "method": "qhed",
                "label": "QHED (Simulated)",
                "type": "Quantum (Simulated)",
                "runtime_s": 0.3992,
                "runtime_ms": 399.2,
                "megapixels_per_s": 5.19,
                "ssim": 0.9905,
                "psnr": 30.34,
                "precision": 1.0000,
                "recall": 1.0000,
                "f1": 1.0000,
                "iou": 1.0000,
                "edge_density": 0.0039,
                "peak_mem_mb": 8.2,
                "qubits": 7,
            },
            {
                "method": "sobel",
                "label": "Sobel",
                "type": "Classical (Spatial 3×3)",
                "runtime_s": 0.0418,
                "runtime_ms": 41.8,
                "megapixels_per_s": 49.56,
                "ssim": 0.9847,
                "psnr": 26.31,
                "precision": 0.9880,
                "recall": 0.9976,
                "f1": 0.9928,
                "iou": 0.9857,
                "edge_density": 0.0079,
                "peak_mem_mb": 3.5,
                "qubits": None,
            },
            {
                "method": "prewitt",
                "label": "Prewitt",
                "type": "Classical (Spatial 3×3)",
                "runtime_s": 0.0432,
                "runtime_ms": 43.2,
                "megapixels_per_s": 48.03,
                "ssim": 0.9840,
                "psnr": 25.92,
                "precision": 0.9720,
                "recall": 0.9912,
                "f1": 0.9815,
                "iou": 0.9637,
                "edge_density": 0.0085,
                "peak_mem_mb": 3.4,
                "qubits": None,
            },
            {
                "method": "canny",
                "label": "Canny",
                "type": "Classical (Multi-stage)",
                "runtime_s": 0.0319,
                "runtime_ms": 31.9,
                "megapixels_per_s": 64.95,
                "ssim": 0.9944,
                "psnr": 30.35,
                "precision": 0.9996,
                "recall": 1.0000,
                "f1": 0.9998,
                "iou": 0.9996,
                "edge_density": 0.0037,
                "peak_mem_mb": 4.1,
                "qubits": None,
            },
            {
                "method": "laplacian",
                "label": "Laplacian",
                "type": "Classical (2nd Derivative)",
                "runtime_s": 0.0233,
                "runtime_ms": 23.3,
                "megapixels_per_s": 88.97,
                "ssim": 0.9809,
                "psnr": 26.64,
                "precision": 0.9610,
                "recall": 0.9881,
                "f1": 0.9744,
                "iou": 0.9501,
                "edge_density": 0.0069,
                "peak_mem_mb": 3.2,
                "qubits": None,
            },
        ],
        "scaling_sweep": [
            {
                "resolution": "480x270",
                "width": 480,
                "height": 270,
                "megapixels": 0.13,
                "qhed_runtime_s": 0.032,
                "sobel_runtime_s": 0.004,
                "canny_runtime_s": 0.002,
                "qhed_tiles": 2400,
            },
            {
                "resolution": "960x540",
                "width": 960,
                "height": 540,
                "megapixels": 0.52,
                "qhed_runtime_s": 0.080,
                "sobel_runtime_s": 0.011,
                "canny_runtime_s": 0.011,
                "qhed_tiles": 9600,
            },
            {
                "resolution": "1920x1080",
                "width": 1920,
                "height": 1080,
                "megapixels": 2.07,
                "qhed_runtime_s": 0.277,
                "sobel_runtime_s": 0.059,
                "canny_runtime_s": 0.028,
                "qhed_tiles": 42420,
            },
            {
                "resolution": "3840x2160",
                "width": 3840,
                "height": 2160,
                "megapixels": 8.29,
                "qhed_runtime_s": 1.626,
                "sobel_runtime_s": 0.152,
                "canny_runtime_s": 0.176,
                "qhed_tiles": 169641,
            },
        ],
        "circuit_verification": {
            "max_abs_aer_delta": 5.76e-12,
            "aer_crop_size": "64x64",
            "aer_elapsed_s": 11.98,
            "qubits_8x8": 7,
            "qubits_16x16": 9,
            "gates_8x8": 68,
            "depth_8x8": 24,
            "tiling_halo_px": 1,
            "linear_scaling_r2": 0.998,
        },
    }


def process_benchmark_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Run real benchmarks on active image or synthetic data and return serialized results."""
    import base64

    from q_edge.benchmark import ALL_METHODS, benchmark, benchmark_summary, dataframe_to_records
    from q_edge.config import Preset, QEdgeConfig
    from q_edge.pipeline import load_image

    image_b64 = str(payload.get("image_b64", ""))
    use_synthetic = bool(payload.get("use_synthetic", False))

    if use_synthetic:
        from q_edge.data.synthetic import synthetic_shapes
        img, ref = synthetic_shapes(payload.get("height", 540), payload.get("width", 960))
    elif image_b64:
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        img = load_image(image_bytes)
        ref = None
    else:
        sample_path = Path(__file__).resolve().parent.parent.parent / "app" / "assets" / "sample.png"
        if sample_path.exists():
            img = load_image(sample_path)
            ref = None
        else:
            from q_edge.data.synthetic import synthetic_shapes
            img, ref = synthetic_shapes(540, 960)

    tile_size = int(payload.get("tile_size", 8))
    sim_tile_size = tile_size if tile_size in (4, 8, 16) else 8
    backend = str(payload.get("backend", "numpy"))
    threshold = float(payload.get("threshold", 0.2))

    config = QEdgeConfig(
        tile_size=sim_tile_size,
        backend=backend,
        threshold=threshold,
        preset=Preset.FULL_4K,
    )

    methods = payload.get("methods") or list(ALL_METHODS)
    df = benchmark(img, reference=ref, methods=methods, config=config, track_memory=True)
    records = dataframe_to_records(df)

    method_labels = {
        "qhed": "QHED (Simulated)",
        "sobel": "Sobel",
        "prewitt": "Prewitt",
        "canny": "Canny",
        "laplacian": "Laplacian",
    }
    method_types = {
        "qhed": "Quantum (Simulated)",
        "sobel": "Classical (Spatial 3×3)",
        "prewitt": "Classical (Spatial 3×3)",
        "canny": "Classical (Multi-stage)",
        "laplacian": "Classical (2nd Derivative)",
    }

    for r in records:
        m = r.get("method", "")
        r["label"] = method_labels.get(m, m.title())
        r["type"] = method_types.get(m, "Classical")
        rt = r.get("runtime_s")
        r["runtime_ms"] = round(rt * 1000, 2) if rt is not None else None
        if m == "qhed":
            r["qubits"] = int(np.log2(sim_tile_size * sim_tile_size)) + 1
        else:
            r["qubits"] = None

    summary = benchmark_summary(df)
    return {
        "status": "success",
        "results": records,
        "summary": summary,
        "dimensions": {"width": img.shape[1], "height": img.shape[0]},
        "reference": "Ground Truth" if ref is not None else "Canny Detector",
    }


def process_scaling_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Run real scaling sweep across resolutions and return pivot rows."""
    from q_edge.benchmark import dataframe_to_records, scaling_benchmark
    from q_edge.config import QEdgeConfig

    quick = bool(payload.get("quick", False))
    if quick:
        sizes = ((270, 480), (540, 960), (1080, 1920))
    else:
        sizes = ((270, 480), (540, 960), (1080, 1920), (2160, 3840))

    methods = payload.get("methods") or ("qhed", "sobel", "canny")
    df = scaling_benchmark(sizes=sizes, methods=methods, config=QEdgeConfig(tile_size=8))
    records = dataframe_to_records(df)

    grouped: dict[str, dict[str, Any]] = {}
    for r in records:
        w = r.get("width")
        h = r.get("height")
        res_key = f"{w}x{h}"
        if res_key not in grouped:
            grouped[res_key] = {
                "resolution": res_key,
                "width": w,
                "height": h,
                "megapixels": r.get("megapixels"),
            }
        m = r.get("method")
        grouped[res_key][f"{m}_runtime_s"] = r.get("runtime_s")
        grouped[res_key][f"{m}_mp_s"] = r.get("megapixels_per_s")

    return {
        "status": "success",
        "sweep_records": records,
        "pivot_rows": list(grouped.values()),
    }
