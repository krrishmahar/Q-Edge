"""Generate the bundled sample image ``app/assets/sample.png``.

The sample is synthetic (no downloads, no licensing questions): the ground-truth shapes
scene with a soft lighting gradient and mild sensor noise, so it looks less like a test
pattern and exercises the detectors on non-flat regions.

Usage: ``uv run python scripts/make_sample.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from q_edge.data.synthetic import synthetic_shapes

OUTPUT = Path(__file__).resolve().parent.parent / "app" / "assets" / "sample.png"


def make_sample(height: int = 720, width: int = 1280, seed: int = 7) -> np.ndarray:
    """Return the sample image as an 8-bit grayscale array."""
    image, _ = synthetic_shapes(height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    lighting = 0.12 * (xx / width) + 0.08 * (yy / height)
    noise = np.random.default_rng(seed).normal(0.0, 0.012, image.shape)
    composed = np.clip(image + lighting + noise, 0.0, 1.0)
    return (composed * 255.0).round().astype(np.uint8)


def main() -> None:
    """Write the sample PNG next to the Streamlit app."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_sample()).save(OUTPUT, optimize=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
