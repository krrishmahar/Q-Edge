"""Q-Edge Streamlit app: tiled Quantum Hadamard Edge Detection beside classical detectors.

Run with ``uv run streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from q_edge.backends import NotConfiguredError, available_backends
from q_edge.benchmark import benchmark, scaling_benchmark
from q_edge.charts import (
    METHOD_LABELS,
    difference_heatmap,
    f1_by_method,
    runtime_vs_resolution,
)
from q_edge.classical import CLASSICAL_METHODS
from q_edge.config import ExecutorKind, Preset, QEdgeConfig
from q_edge.pipeline import PipelineResult, preprocess, run_pipeline
from q_edge.quantum.encoding import SUPPORTED_TILE_SIZES
from q_edge.scheduler.executor import ProgressCallback
from q_edge.ui import (
    HEATMAP_MAX_SIDE,
    UploadError,
    decode_upload,
    downscale,
    prepare_for_backend,
    preview,
    signed_difference,
    to_png_bytes,
)

SAMPLE_PATH = Path(__file__).resolve().parent / "assets" / "sample.png"

PRESET_LABELS = {
    Preset.FAST: "Fast (512 px)",
    Preset.BALANCED: "Balanced (1280 px)",
    Preset.FULL_4K: "Full-4K (3840 px)",
}

BACKEND_HELP = (
    "numpy: exact batched statevector simulation (fast). "
    "aer: real Qiskit circuits on Aer, one per tile (small images only). "
    "gpu: CuPy if a CUDA GPU is present, otherwise NumPy. "
    "ibm-hardware: interface stub, not configured."
)

st.set_page_config(page_title="Q-Edge", page_icon=":material/blur_on:", layout="wide")


# --- cached computation -------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def load_sample() -> bytes:
    """Read the bundled sample image."""
    return SAMPLE_PATH.read_bytes()


def compute_quantum(
    data: bytes, settings: dict[str, Any], progress: ProgressCallback | None = None
) -> tuple[PipelineResult, str | None]:
    """Decode, apply backend limits and run the tiled QHED pipeline (no Streamlit calls)."""
    config = QEdgeConfig(**settings)
    image = preprocess(decode_upload(data).image, config.preset.max_side)
    image, config, note = prepare_for_backend(image, config)
    return run_pipeline(image, config, progress=progress), note


@st.cache_data(show_spinner=False, max_entries=8)
def run_quantum(data: bytes, settings: dict[str, Any]) -> tuple[PipelineResult, str | None]:
    """Cached QHED run with a progress bar and tile counter (shown on a cache miss)."""
    bar = st.progress(0.0, text="Starting...")

    def report(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Processing tiles: {done:,} / {total:,}")

    try:
        return compute_quantum(data, settings, report)
    finally:
        bar.empty()


@st.cache_data(show_spinner=False, max_entries=8)
def run_benchmark(data: bytes, settings: dict[str, Any]) -> pd.DataFrame:
    """Metrics table for all methods on the processed image (Canny as the reference)."""
    result, _ = compute_quantum(data, {**settings, "backend": "numpy"})
    return benchmark(result.image, config=result.config)


@st.cache_data(show_spinner=False, max_entries=2)
def run_scaling() -> pd.DataFrame:
    """Runtime versus resolution sweep (synthetic scenes, NumPy backend)."""
    return scaling_benchmark(methods=tuple(METHOD_LABELS))


# --- sidebar ------------------------------------------------------------------------------


def sidebar() -> tuple[bytes, str, dict[str, Any], str, bool]:
    """Render the controls; return image bytes, its name, settings, method and view mode."""
    st.sidebar.title("Q-Edge")
    st.sidebar.caption("Hybrid quantum-classical edge detection (QHED)")

    upload = st.sidebar.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"], help="PNG or JPEG, up to 3840x2160."
    )
    if upload is not None:
        data = upload.getvalue()
        name = upload.name
    else:
        data, name = load_sample(), "sample.png (bundled)"

    preset = st.sidebar.radio(
        "Resolution preset",
        list(Preset),
        index=1,
        format_func=lambda p: PRESET_LABELS[Preset(p)],
        help="Images are downscaled so the longest side fits the preset.",
    )
    tile_size = st.sidebar.select_slider(
        "Tile size",
        options=list(SUPPORTED_TILE_SIZES),
        value=8,
        format_func=lambda t: f"{t}x{t} ({int(math.log2(t * t)) + 1} qubits)",
    )
    backends = [b for b in ("numpy", "aer", "gpu", "ibm-hardware") if b in available_backends()]
    backend = st.sidebar.selectbox("Backend", backends, index=0, help=BACKEND_HELP)
    shots: int | None = None
    if backend in ("aer", "ibm-hardware"):
        exact = st.sidebar.checkbox("Exact statevector (no shot noise)", value=True)
        if not exact:
            shots = int(st.sidebar.number_input("Shots per circuit", 64, 1_000_000, 4096, step=512))
    workers = st.sidebar.slider("Workers (0 = auto)", 0, 16, 0)
    threshold = st.sidebar.slider("Edge threshold", 0.0, 1.0, 0.2, 0.01)

    st.sidebar.divider()
    method = st.sidebar.selectbox(
        "Classical comparison",
        list(CLASSICAL_METHODS),
        index=list(CLASSICAL_METHODS).index("sobel"),
        format_func=lambda m: METHOD_LABELS.get(m, m),
    )
    binary = st.sidebar.toggle("Show thresholded (binary) maps", value=False)

    settings: dict[str, Any] = {
        "tile_size": int(tile_size),
        "backend": backend,
        "workers": int(workers),
        "shots": shots,
        "threshold": float(threshold),
        "preset": Preset(preset or Preset.BALANCED).value,
        "executor": ExecutorKind.THREAD.value,
        "seed": 7,
    }
    return data, name, settings, method, binary


# --- tabs ---------------------------------------------------------------------------------


def results_tab(
    data: bytes, name: str, settings: dict[str, Any], method: str, binary: bool
) -> None:
    """Original vs. QHED vs. classical, difference heatmap, metrics and download."""
    try:
        result, note = run_quantum(data, settings)
    except (UploadError, NotConfiguredError) as exc:
        st.error(str(exc))
        return
    if note:
        st.info(note)

    cfg = result.config
    h, w = result.image.shape
    qubits = int(math.log2(cfg.tile_size**2)) + 1
    cols = st.columns(5)
    cols[0].metric("Processed size", f"{w}x{h}")
    cols[1].metric("Tiles", f"{result.num_tiles:,}")
    cols[2].metric("Qubits per circuit", qubits)
    cols[3].metric("QHED time", f"{result.total_time:.2f} s")
    cols[4].metric("Throughput", f"{result.tiles_per_second:,.0f} tiles/s")

    classical = CLASSICAL_METHODS[method](result.image)
    quantum_view = result.binary if binary else result.edges
    classical_view = classical >= max(cfg.threshold, 1e-12) if binary else classical
    label = METHOD_LABELS[method]

    left, middle, right = st.columns(3)
    left.image(preview(result.image), caption=f"Original: {name}", width="stretch")
    middle.image(preview(quantum_view), caption=f"QHED, {cfg.backend} backend", width="stretch")
    right.image(preview(classical_view), caption=label, width="stretch")

    diff = signed_difference(quantum_view.astype(float), classical_view.astype(float))
    st.plotly_chart(difference_heatmap(downscale(diff, HEATMAP_MAX_SIDE), label), width="stretch")
    st.caption(
        "Orange: QHED responds more strongly; blue: the classical detector does. QHED uses "
        "1-pixel forward differences, so it produces thinner edges than 3x3 operators."
    )

    st.download_button(
        "Download edge map (PNG)",
        data=to_png_bytes(quantum_view),
        file_name="q-edge-edges.png",
        mime="image/png",
        type="primary",
    )

    st.subheader("Metrics")
    if cfg.backend != "numpy":
        st.caption("Metrics are computed with the NumPy backend (identical results, faster).")
    with st.spinner("Scoring all methods..."):
        table = run_benchmark(data, settings)
    st.dataframe(
        table.style.format(
            {
                "runtime_s": "{:.4f}",
                "tiles_per_s": "{:,.0f}",
                "megapixels_per_s": "{:.1f}",
                "ssim": "{:.3f}",
                "psnr": "{:.2f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "edge_density": "{:.4f}",
                "peak_mem_mb": "{:.1f}",
            },
            na_rep="-",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Uploaded photos have no ground truth, so Canny is the reference edge map (Canny "
        "therefore scores F1 = 1). Precision/recall allow a 1-pixel tolerance."
    )
    st.plotly_chart(f1_by_method(table), width="stretch")


def benchmarks_tab() -> None:
    """Runtime-vs-resolution sweep with an honest performance note."""
    st.warning(
        "Honest note: QHED here runs on a classical simulator. It gives **no speedup** over "
        "classical filters (OpenCV is 5-10x faster). What the chart shows is the scaling "
        "trend: tiling keeps the circuit size fixed, so runtime grows linearly with pixels."
    )
    if st.button("Run scaling benchmark (480p to 4K)"):
        with st.spinner("Benchmarking 4 resolutions x 5 methods..."):
            st.session_state["scaling"] = run_scaling()
    scaling = st.session_state.get("scaling")
    if scaling is None:
        st.info("Press the button to measure runtime on synthetic scenes up to 3840x2160.")
        return
    st.plotly_chart(runtime_vs_resolution(scaling), width="stretch")
    st.dataframe(
        scaling[["width", "height", "method", "runtime_s", "megapixels_per_s", "f1"]],
        hide_index=True,
        width="stretch",
    )


ARCHITECTURE_DOT = """
digraph qedge {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded", fontname="sans-serif", fontsize=11];
  edge [fontname="sans-serif", fontsize=9];
  upload [label="Image\\n(PNG/JPEG up to 4K)"];
  pre [label="Preprocess\\ngrayscale + preset resize"];
  tiler [label="Tiler\\nT x T tiles, 1px halo"];
  sched [label="Scheduler\\nauto workers / batch size"];
  subgraph cluster_backends {
    label="Backends (one Protocol: run_tiles)"; style="rounded,dashed";
    numpy [label="NumPy\\nexact statevector"];
    aer [label="Qiskit Aer\\nreal circuits"];
    gpu [label="GPU (CuPy)\\nfallback: NumPy"];
    hw [label="IBM hardware\\n(stub)"];
  }
  stitch [label="Stitch cores\\nseam-free"];
  post [label="Normalise + threshold"];
  out [label="Edge map + metrics"];
  classical [label="Classical baselines\\nSobel / Prewitt / Canny / Laplacian"];
  upload -> pre -> tiler -> sched;
  sched -> numpy; sched -> aer; sched -> gpu; sched -> hw;
  numpy -> stitch; aer -> stitch; gpu -> stitch; hw -> stitch;
  stitch -> post -> out;
  pre -> classical -> out;
}
"""


def scalability_tab() -> None:
    """Architecture diagram, resolution independence and roadmap."""
    st.subheader("Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    st.markdown(
        "Every tile is an independent QHED circuit on `log2(T^2) + 1` qubits. Image size "
        "only changes **how many** circuits run, never **how big** they are, and tiles are "
        "processed in parallel batches."
    )
    rows = []
    for width, height in ((640, 360), (1280, 720), (1920, 1080), (3840, 2160)):
        row: dict[str, Any] = {"resolution": f"{width}x{height}"}
        for t in SUPPORTED_TILE_SIZES:
            tiles = math.ceil(height / (t - 1)) * math.ceil(width / (t - 1))
            row[f"tiles @ {t}x{t} ({int(math.log2(t * t)) + 1} qubits)"] = f"{tiles:,}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Roadmap")
    st.markdown(
        """
1. **Hardware execution.** Implement the IBM Runtime backend (SamplerV2 in batch mode,
   transpilation per device) behind the existing `Backend` protocol.
2. **Noise mitigation.** Readout-error mitigation, dynamical decoupling and zero-noise
   extrapolation, and cheaper state preparation for large tiles.
3. **Real-time video.** Reuse the tile pipeline per frame, update only changed tiles,
   and stream frames through the GPU backend.
4. **Distributed execution.** Ship tile batches to a cluster (Ray/Dask) or to several QPUs;
   tiles are independent, so the scheduler scales horizontally.
"""
    )


# --- main ---------------------------------------------------------------------------------


def main() -> None:
    """Render the app."""
    data, name, settings, method, binary = sidebar()
    st.title("Q-Edge: quantum Hadamard edge detection")
    st.caption(
        "A prototype that shows the architecture and correctness of tiled QHED. It does not "
        "claim quantum advantage: all quantum results here come from classical simulation."
    )
    results, benchmarks, scalability = st.tabs(["Results", "Benchmarks", "Scalability"])
    with results:
        results_tab(data, name, settings, method, binary)
    with benchmarks:
        benchmarks_tab()
    with scalability:
        scalability_tab()


main()
