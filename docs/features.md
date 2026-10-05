# Features

Every feature listed here is implemented, tested and merged on `main`. The last column links to
the code.

## Quantum core

| Feature | Details | Code |
|---|---|---|
| Amplitude encoding | Row-major flattening, L2 normalisation, norm kept for rescaling; `StatePreparation` circuit | [`quantum/encoding.py`](../src/q_edge/quantum/encoding.py) |
| Zero-tile safety | All-black tiles get zero amplitudes and norm 0, never NaN; the circuit path skips them | `normalize_tiles` |
| QHED circuit | Ancilla in `\|+>`, multi-controlled-X decrement on `n+1` qubits, Hadamard on the ancilla | [`quantum/qhed_circuit.py`](../src/q_edge/quantum/qhed_circuit.py) |
| Aer execution | Exact statevector mode (signed differences) or shot-based mode (magnitudes from probabilities) | `circuit_differences` |
| Fast path | Exact NumPy simulation of the same unitary, vectorised over thousands of tiles | [`quantum/qhed_fast.py`](../src/q_edge/quantum/qhed_fast.py) |
| Bidirectional edges | Horizontal pass on the tile, vertical pass on the transpose, combined as `sqrt(h^2 + v^2)` | `oriented_responses`, `combine` |
| Proven equivalence | Tests check fast against circuit (atol 1e-8), the decrement unitary against a cyclic shift, and shot estimates against exact values | [`tests/test_quantum.py`](../tests/test_quantum.py) |
| Bounded circuits | 4x4, 8x8 and 16x16 tiles use 5, 7 and 9 qubits; one giant circuit is never built | `SUPPORTED_TILE_SIZES` |

## Tiling and pipeline

| Feature | Details | Code |
|---|---|---|
| Halo tiling | `T x T` tiles at stride `T-1`, so neighbouring tiles share a 1-pixel border | [`tiling/tiler.py`](../src/q_edge/tiling/tiler.py) |
| Seam-free stitching | Cores of `(T-1) x (T-1)` tile the image exactly; the result equals the untiled computation | `stitch` |
| Any image size | Edge padding handles odd sizes and sizes that are not multiples of the tile, from 1x1 to 4K | `split` |
| Lazy batches | Tiles come from a strided view and are copied one batch at a time | `TileGrid.batch` |
| Flexible loading | Path, bytes, PIL image or NumPy array; 8/16-bit, RGB(A), CMYK; EXIF orientation; alpha flattened onto white | [`pipeline.py`](../src/q_edge/pipeline.py) `load_image` |
| Resolution presets | Fast 512, Balanced 1280, Full-4K 3840 (longest side), area interpolation | [`config.py`](../src/q_edge/config.py) |
| Validated configuration | Frozen dataclass; invalid tile size, shots, threshold, workers or preset raise `ValueError` | `QEdgeConfig` |
| Stage timings | Load, preprocess, tile, compute, stitch and postprocess timed separately | `PipelineResult.timings` |
| Progress reporting | `(done, total)` callback after each batch; drives the UI progress bar | `run_pipeline(progress=...)` |

## Backends and scheduling

| Feature | Details | Code |
|---|---|---|
| Backend protocol | One method, `run_tiles(batch) -> responses`; backends are looked up by name | [`backends/base.py`](../src/q_edge/backends/base.py) |
| Plug-in registry | `register_backend("name", factory)`; a test adds a custom backend without touching the pipeline | `register_backend` |
| NumPy backend | Default; exact and fast | [`numpy_backend.py`](../src/q_edge/backends/numpy_backend.py) |
| Aer backend | Real circuits; `shots` and `seed` taken from the config | [`aer_backend.py`](../src/q_edge/backends/aer_backend.py) |
| GPU backend | CuPy when a CUDA device is present, otherwise a logged fallback to NumPy; picklable | [`gpu_backend.py`](../src/q_edge/backends/gpu_backend.py) |
| Hardware stub | IBM Runtime placeholder that raises `NotConfiguredError` with setup instructions; no network | [`hardware_stub.py`](../src/q_edge/backends/hardware_stub.py) |
| Parallel executor | Thread pool (default), process pool or serial; chunked batches; bounded in-flight work; order-preserving | [`scheduler/executor.py`](../src/q_edge/scheduler/executor.py) |
| Resource awareness | Detects CPU count, RAM and GPU; picks tile size, workers and batch size automatically | [`scheduler/resources.py`](../src/q_edge/scheduler/resources.py) |

## Classical baselines and metrics

| Feature | Details | Code |
|---|---|---|
| Sobel, Prewitt, Laplacian | OpenCV, replicate borders, normalised magnitude | [`classical/`](../src/q_edge/classical/) |
| Canny | OpenCV with Gaussian pre-blur and median-based automatic thresholds (with a floor and clamped to 255) | [`classical/canny.py`](../src/q_edge/classical/canny.py) |
| Quality metrics | SSIM (adaptive window for tiny images), PSNR (`inf` when identical), precision/recall/F1 with Euclidean pixel tolerance, edge density | [`metrics/quality.py`](../src/q_edge/metrics/quality.py) |
| Performance metrics | Wall time, throughput and peak memory via `tracemalloc` | [`metrics/perf.py`](../src/q_edge/metrics/perf.py) |
| `benchmark()` | One-image comparison DataFrame: method, runtime_s, tiles_per_s, megapixels_per_s, ssim, psnr, precision, recall, f1, edge_density, peak_mem_mb | [`benchmark.py`](../src/q_edge/benchmark.py) |
| `scaling_benchmark()` | Runtime against resolution (480p to 4K) on synthetic scenes with ground truth | `scaling_benchmark` |
| Synthetic data | Resolution-independent shapes scene with exact ground-truth boundaries | [`data/synthetic.py`](../src/q_edge/data/synthetic.py) |

## Streamlit app

| Feature | Details |
|---|---|
| Upload or sample | PNG/JPEG up to 4K; the bundled synthetic sample loads when nothing is uploaded |
| Full control sidebar | Preset, tile size with qubit count, backend, shots, workers, threshold, classical method, binary toggle |
| Side-by-side view | Original, QHED and classical, with downsized previews for large images |
| Difference heatmap | Signed `QHED - classical` on a diverging blue-grey-orange scale, downsampled for responsiveness |
| Metrics table and F1 chart | All five methods, with the reference and the tolerance explained |
| Runtime vs resolution chart | On demand; each method keeps the same colour across charts |
| Scalability tab | Graphviz architecture diagram, tiles-per-resolution table, roadmap |
| Caching | `st.cache_data` keyed on image bytes and settings |
| Progress bar | Tile counter during processing |
| Download | Full-resolution edge map as PNG |
| Safety limits | Circuit backends are capped at 64 px and 4x4 tiles, with an explanatory note |
| Error handling | Upload, decode and hardware-configuration errors appear as messages, never as stack traces |
| Privacy | Binds to `localhost`, usage statistics off, no external-IP lookup, no runtime network calls |

## Hardening and quality

- Upload validation: empty, larger than 50 MB, corrupt, truncated, non-PNG/JPEG, larger than
  3840x2160, smaller than 2 px, and decompression bombs.
- Numerical noise floor: flat images produce all-zero edge maps instead of amplified rounding
  noise.
- 165 tests, with coverage at about 98% and a gate that fails under 80%.
- ruff lint and format checks, and strict mypy on `src` and `app`.
- An end-to-end demo script and a clean-clone verification.

## Known limitations

- The quantum path is simulated, so it shows no speedup (OpenCV is 5-10x faster).
- The synthetic ground truth uses the forward-difference convention, which favours QHED.
- The GPU backend has not been run on real CUDA hardware.
- The IBM hardware backend is a stub.
- The metrics table shows `None` instead of `-` under `tiles_per_s` for classical methods
  (cosmetic).
