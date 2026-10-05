# API reference

All paths are relative to the repository root. Arrays are NumPy `float64` unless stated.
Images are 2-D grayscale in `[0, 1]`. Tile batches have shape `(B, T, T)`, and per-tile cores
have shape `(B, T-1, T-1)`.

## Quick index

| Import | Purpose | File |
|---|---|---|
| `q_edge.pipeline.run_pipeline` | Run QHED end to end on an image | `src/q_edge/pipeline.py` |
| `q_edge.config.QEdgeConfig` | Run settings | `src/q_edge/config.py` |
| `q_edge.backends.get_backend` / `register_backend` | Backend registry | `src/q_edge/backends/base.py` |
| `q_edge.benchmark.benchmark` / `scaling_benchmark` | Comparison DataFrames | `src/q_edge/benchmark.py` |
| `q_edge.quantum.qhed_fast.qhed_edge_magnitude` | Fast QHED on tile batches | `src/q_edge/quantum/qhed_fast.py` |
| `q_edge.quantum.qhed_circuit.build_qhed_circuit` | Real Qiskit circuit for one tile | `src/q_edge/quantum/qhed_circuit.py` |
| `q_edge.tiling.tiler.split` / `stitch` | Halo tiling | `src/q_edge/tiling/tiler.py` |
| `q_edge.scheduler.executor.execute` | Parallel tile execution | `src/q_edge/scheduler/executor.py` |

---

## `q_edge.config` (`src/q_edge/config.py`)

### `class Preset(StrEnum)`
`FAST = "fast"` (512 px), `BALANCED = "balanced"` (1280 px), `FULL_4K = "full-4k"` (3840 px).
The `.max_side` property returns the longest processed side in pixels.

### `class ExecutorKind(StrEnum)`
`THREAD = "thread"` (default), `PROCESS = "process"`, `SERIAL = "serial"`.

### `@dataclass(frozen=True) class QEdgeConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tile_size` | `int` | `8` | One of `4, 8, 16` |
| `backend` | `str` | `"numpy"` | Registered backend name |
| `workers` | `int` | `0` | `0` = auto |
| `batch_size` | `int` | `0` | Tiles per task; `0` = auto |
| `shots` | `int \| None` | `None` | `None` = exact statevector |
| `threshold` | `float` | `0.2` | Binarisation threshold in `[0, 1]` |
| `preset` | `Preset` | `BALANCED` | Strings are accepted and converted |
| `seed` | `int \| None` | `None` | Shot-sampling seed |
| `executor` | `ExecutorKind` | `THREAD` | Strings are accepted and converted |

- Invalid values raise `ValueError`.
- `with_updates(**changes) -> QEdgeConfig` returns a validated copy.

---

## `q_edge.pipeline` (`src/q_edge/pipeline.py`)

### `run_pipeline(source, config=None, backend=None, progress=None) -> PipelineResult`
Runs load → preprocess → tile → backend → stitch → normalise → threshold.

- `source`: `str | Path | bytes | PIL.Image.Image | np.ndarray`.
- `backend`: a `Backend` instance; defaults to `get_backend(config.backend, config)`.
- `progress`: `Callable[[int, int], None]`, called with `(tiles_done, tiles_total)`.
- Raises `NotConfiguredError` for the hardware stub, and `ValueError` or `OSError` for bad
  input.

### `@dataclass(frozen=True) class PipelineResult`

| Attribute | Description |
|---|---|
| `image` | Preprocessed grayscale input `(H, W)` |
| `magnitude` | Raw `sqrt(h^2 + v^2)` in intensity units |
| `edges` | `magnitude` normalised to `[0, 1]` |
| `binary` | `edges >= threshold` (bool) |
| `num_tiles`, `workers`, `batch_size` | Values actually used |
| `timings` | `{"load", "preprocess", "tile", "compute", "stitch", "postprocess"}` in seconds |
| `config` | The `QEdgeConfig` used |
| `total_time` *(property)* | Sum of the timings |
| `tiles_per_second` *(property)* | `num_tiles / timings["compute"]` |

### `load_image(source) -> FloatArray`
Decodes to grayscale `[0, 1]`:
- Handles EXIF orientation, 16-bit (divided by 65535), RGB(A) and CMYK.
- Flattens transparency onto white.
- For arrays, integer dtypes are scaled by their maximum.

### `preprocess(image, max_side) -> FloatArray`
Area-downscales so `max(H, W) <= max_side`. Returns the input unchanged when it already fits.

---

## `q_edge.quantum` (`src/q_edge/quantum/`)

### `encoding.py`

| Name | Signature | Notes |
|---|---|---|
| `SUPPORTED_TILE_SIZES` | `(4, 8, 16)` | |
| `num_data_qubits` | `(num_pixels: int) -> int` | `ValueError` unless the count is a power of two ≥ 2 |
| `normalize_tiles` | `(tiles) -> (amplitudes (B, N), norms (B,))` | Zero tiles give zeros, never NaN |
| `amplitude_encoding_circuit` | `(amplitudes) -> QuantumCircuit` | `StatePreparation`; rejects non-unit vectors |

### `qhed_fast.py`

| Name | Signature | Notes |
|---|---|---|
| `qhed_statevector` | `(amplitudes (B, N)) -> (B, 2N)` | Final QHED statevectors |
| `qhed_differences` | `(flat_tiles (B, N)) -> (B, N)` | Cyclic `c_i - c_{i+1}`, rescaled |
| `oriented_responses` | `(tiles, differences_fn) -> (h (B,T,T-1), v (B,T-1,T))` | Shared by the fast and circuit paths |
| `combine` | `(h, v) -> (B, T-1, T-1)` | `hypot` over the common core |
| `qhed_responses` | `(tiles) -> (h, v)` | |
| `qhed_edge_magnitude` | `(tiles) -> (B, T-1, T-1)` | Used by `NumpyBackend` |
| `AMPLITUDE_TO_DIFFERENCE` | `2.0` | Amplitude-to-difference scale factor |

### `qhed_circuit.py`

| Name | Signature | Notes |
|---|---|---|
| `decrement_circuit` | `(num_qubits) -> QuantumCircuit` | `\|k> -> \|k-1 mod 2^n>` from MCX gates |
| `build_qhed_circuit` | `(amplitudes, measure=False) -> QuantumCircuit` | `n+1` qubits; qubit 0 is the ancilla |
| `circuit_differences` | `(flat_tiles, shots=None, seed=None) -> (B, N)` | Runs on `AerSimulator`; shot mode returns magnitudes |
| `circuit_responses` | `(tiles, shots=None, seed=None) -> (h, v)` | |
| `circuit_edge_magnitude` | `(tiles, shots=None, seed=None) -> (B, T-1, T-1)` | Used by `AerBackend` |

```python
import numpy as np
from q_edge.quantum.encoding import normalize_tiles
from q_edge.quantum.qhed_circuit import build_qhed_circuit

amps, _ = normalize_tiles(np.random.rand(1, 4, 4))
print(build_qhed_circuit(amps[0]).draw())  # 5-qubit QHED circuit
```

---

## `q_edge.tiling.tiler` (`src/q_edge/tiling/tiler.py`)

| Name | Signature | Notes |
|---|---|---|
| `split` | `(image (H, W), tile_size) -> TileGrid` | Edge-padded, stride `T-1` |
| `stitch` | `(cores (num_tiles, T-1, T-1), grid) -> (H, W)` | Exact inverse placement |
| `TileGrid.rows`, `.cols`, `.num_tiles`, `.core_size` | properties | |
| `TileGrid.batch` | `(start, stop) -> (k, T, T)` | Contiguous copy, row-major tile order |
| `TileGrid.iter_batches` | `(batch_size) -> Iterator[(start, stop)]` | |
| `TileGrid.all_tiles` | `() -> (num_tiles, T, T)` | |

```python
from q_edge.tiling.tiler import split, stitch
from q_edge.quantum.qhed_fast import qhed_edge_magnitude

grid = split(image, 8)
edges = stitch(qhed_edge_magnitude(grid.all_tiles()), grid)  # same shape as image
```

---

## `q_edge.backends` (`src/q_edge/backends/`)

### Protocol (`base.py`)

```python
class Backend(Protocol):
    @property
    def name(self) -> str: ...
    def run_tiles(self, batch: FloatArray) -> FloatArray: ...  # (B,T,T) -> (B,T-1,T-1)
```

| Function | Signature | Notes |
|---|---|---|
| `register_backend` | `(name, factory: Callable[[QEdgeConfig], Backend], *, overwrite=False)` | Names are case-insensitive |
| `available_backends` | `() -> list[str]` | |
| `get_backend` | `(name, config=None) -> Backend` | `ValueError` that lists the available names |

### Implementations

| Class | File | Constructor | Registered as |
|---|---|---|---|
| `NumpyBackend` | `numpy_backend.py` | `NumpyBackend()` | `numpy` |
| `AerBackend` | `aer_backend.py` | `AerBackend(shots=None, seed=None)` | `aer` |
| `GpuBackend` | `gpu_backend.py` | `GpuBackend()`; `.uses_gpu` property | `gpu` |
| `IBMHardwareBackend` | `hardware_stub.py` | `IBMHardwareBackend(shots=4096)`; `run_tiles` raises `NotConfiguredError` | `ibm-hardware` |

`gpu_backend.py` also exports `load_cupy() -> module | None` and
`qhed_edge_magnitude_xp(tiles, xp)`, an array-module-generic implementation.

```python
from dataclasses import dataclass
from q_edge.backends import register_backend
from q_edge.pipeline import run_pipeline
from q_edge.config import QEdgeConfig


@dataclass(frozen=True)
class MyBackend:
    name: str = "mine"

    def run_tiles(self, batch): ...  # return (B, T-1, T-1)


register_backend("mine", lambda cfg: MyBackend())
run_pipeline("photo.png", QEdgeConfig(backend="mine"))  # no pipeline change needed
```

---

## `q_edge.scheduler` (`src/q_edge/scheduler/`)

| Name | File | Signature / fields |
|---|---|---|
| `execute` | `executor.py` | `(grid, backend, *, workers=1, batch_size=4096, kind=ExecutorKind.THREAD, progress=None) -> cores` |
| `SystemResources` | `resources.py` | `cpu_count, total_memory, available_memory, gpu_available` |
| `AutoSettings` | `resources.py` | `tile_size, workers, batch_size` |
| `detect_memory` | `resources.py` | `() -> (total, available)` bytes, with a fallback |
| `detect_resources` | `resources.py` | `() -> SystemResources` |
| `auto_settings` | `resources.py` | `(image_shape, backend="numpy", resources=None) -> AutoSettings` |

---

## `q_edge.classical` (`src/q_edge/classical/`)

| Function | File | Signature |
|---|---|---|
| `sobel` | `sobel.py` | `(image, ksize=3) -> [0,1] map` |
| `prewitt` | `prewitt.py` | `(image) -> [0,1] map` |
| `laplacian` | `laplacian.py` | `(image, ksize=3) -> [0,1] map` |
| `canny` | `canny.py` | `(image, low=None, high=None, blur=3) -> {0,1} map` |
| `auto_thresholds` | `canny.py` | `(image_u8, sigma=0.33) -> (low, high)` |
| `CLASSICAL_METHODS` | `__init__.py` | `{"sobel", "prewitt", "canny", "laplacian"} -> function` |

---

## `q_edge.metrics` (`src/q_edge/metrics/`)

| Name | File | Signature |
|---|---|---|
| `ssim` | `quality.py` | `(prediction, reference) -> float` |
| `psnr` | `quality.py` | `(prediction, reference) -> float` (`inf` if identical) |
| `edge_scores` | `quality.py` | `(prediction_bool, reference_bool, tolerance=1.0) -> EdgeScores(precision, recall, f1)` |
| `edge_density` | `quality.py` | `(binary) -> float` |
| `measure` | `perf.py` | context manager yielding `PerfSample(wall_time, peak_memory, peak_memory_mb)`; `track_memory=True` |
| `throughput` | `perf.py` | `(count, seconds) -> float` |

## `q_edge.benchmark` (`src/q_edge/benchmark.py`)

| Name | Signature |
|---|---|
| `benchmark` | `(source, reference=None, methods=ALL_METHODS, config=None, track_memory=True) -> DataFrame` |
| `scaling_benchmark` | `(sizes=DEFAULT_SCALING_SIZES, methods=("qhed", "sobel", "canny"), config=None) -> DataFrame` |
| `detect_edges` | `(image, method, config) -> (edges, binary, num_tiles)` |
| `ALL_METHODS` | `("qhed", "sobel", "prewitt", "canny", "laplacian")` |
| `BENCHMARK_COLUMNS` | `method, runtime_s, tiles_per_s, megapixels_per_s, ssim, psnr, precision, recall, f1, edge_density, peak_mem_mb` |

```python
from q_edge.benchmark import benchmark
from q_edge.data.synthetic import synthetic_shapes

image, truth = synthetic_shapes(1080, 1920)
print(benchmark(image, truth))  # one row per method
```

---

## Supporting modules

| Module | File | Public API |
|---|---|---|
| `q_edge.imaging` | `src/q_edge/imaging.py` | `normalise(magnitude)`, `NOISE_FLOOR = 1e-9` |
| `q_edge.data.synthetic` | `src/q_edge/data/synthetic.py` | `synthetic_shapes(height=256, width=256) -> (image, edges)`, `label_edges(labels)` |
| `q_edge.ui` | `src/q_edge/ui.py` | `decode_upload(bytes) -> DecodedUpload`, `UploadError`, `prepare_for_backend(image, config)`, `downscale`, `preview`, `to_png_bytes`, `signed_difference`; limits `MAX_UPLOAD_SIDE`, `MAX_UPLOAD_BYTES`, `CIRCUIT_MAX_SIDE`, `PREVIEW_MAX_SIDE`, `HEATMAP_MAX_SIDE` |
| `q_edge.charts` | `src/q_edge/charts.py` | `runtime_vs_resolution(df)`, `f1_by_method(df)`, `difference_heatmap(diff, label)`, `METHOD_COLORS`, `METHOD_LABELS` |

## Entry points and scripts

| Command | File |
|---|---|
| `uv run streamlit run app/streamlit_app.py` | `app/streamlit_app.py` |
| `uv run python scripts/demo.py [--image PATH] [--scaling]` | `scripts/demo.py` |
| `uv run python scripts/make_sample.py` | `scripts/make_sample.py` (regenerates `app/assets/sample.png`) |
| `uv run --with playwright python scripts/capture_screenshots.py --url URL` | `scripts/capture_screenshots.py` |

## Errors

| Exception | Raised by | Meaning |
|---|---|---|
| `ValueError` | config, tiler, metrics, `get_backend` | Invalid argument; the message names the problem |
| `UploadError(ValueError)` | `q_edge.ui.decode_upload` | A user-facing upload problem |
| `NotConfiguredError(RuntimeError)` | `IBMHardwareBackend.run_tiles` | Hardware execution is not set up |
