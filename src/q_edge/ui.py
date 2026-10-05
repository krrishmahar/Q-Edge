"""Framework-independent helpers for the Streamlit front-end.

Everything here is plain Python/NumPy so it can be unit-tested without a browser.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

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
CIRCUIT_MAX_SIDE = 96

#: Circuit-level backends that need the small-image cap.
CIRCUIT_BACKENDS: frozenset[str] = frozenset({"aer", "ibm-hardware"})

#: Longest side of images shown in the browser.
PREVIEW_MAX_SIDE = 1024

#: Longest side of the interactive difference heatmap (one plotly cell per pixel).
HEATMAP_MAX_SIDE = 360


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
