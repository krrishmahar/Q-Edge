# Design Decisions

Defaults chosen where the brief was ambiguous. Each entry: decision, then rationale.

## Tooling
- **Nested git repository.** The project lives in `del/`, which sits inside an unrelated parent
  repository owned by another user. `git init` in `del/` creates an independent repository and
  leaves the parent untouched.
- **uv installed with the official Astral installer** (no pip). The virtualenv is created with
  `uv venv`; every command runs through `uv run`.
- **Task runner: `justfile`.** `just` is installed with `uv tool install rust-just`. `make` is not
  available on the Windows host.
- **Configuration uses frozen dataclasses**, not pydantic, to avoid an extra dependency.

## Algorithm
- **Ancilla is qubit 0 (least-significant bit).** With `|c> (x) |+>`, a cyclic decrement over
  all `n + 1` qubits and a Hadamard on the ancilla, the ancilla-|1> amplitudes equal
  `(c_i - c_{i+1}) / 2`; multiplying by `2 * norm` recovers the raw difference.
- **Sign in shot mode.** Measurement only yields `|c_i - c_{i+1}|`; this is sufficient because
  the edge magnitude is `sqrt(h^2 + v^2)`.
- **Wrap-around terms are discarded.** The cyclic shift also produces row-crossing and
  last-to-first differences; these are masked out after the circuit.

## Tiling
- **Stride `T - 1` with a 1-pixel halo.** Each `T x T` tile yields a `(T - 1) x (T - 1)` core;
  cores tile the image exactly, so stitching is a reshape and the result equals the untiled
  forward-difference gradient (verified by tests). The image is edge-padded on the bottom and
  right, so the last row and column use edge replication.
- **Normalisation.** The stitched magnitude is divided by its global maximum, then thresholded.
- **4K measurement (wave 2).** A synthetic 3840x2160 image with 8x8 tiles (169,641 tiles) runs
  end to end through the fast path in about 1.7 s on the 8-core development machine, single
  process.

## Backends and scheduling
- **Backend registry.** Backends are created by name through `get_backend`; new backends call
  `register_backend` and need no pipeline changes (covered by a test with a custom backend).
- **Thread pool is the default executor.** NumPy releases the GIL in the heavy kernels, and
  threads avoid pickling tiles. A process pool is available (`executor="process"`) but was
  slower on 4K in practice (3.2 s vs 1.1 s for threads, 1.7 s serial, 8x8 tiles, 8 cores)
  because each batch has to be serialised to the worker. Processes also behave poorly inside
  Streamlit on Windows.
- **Bounded in-flight batches** (2 per worker) keep peak memory flat for 4K images.
- **Auto settings.** Batch size targets 2% of available RAM per in-flight batch, clamped to
  256-65,536 tiles. Auto tile size is 16 above 2 MP and 8 otherwise. Circuit backends use 4x4
  tiles (5 qubits) and batches of 32.
- **GPU backend** uses CuPy only if it is installed and sees a CUDA device. Otherwise it logs a
  warning and runs the same math with NumPy. CuPy is deliberately not a dependency.
- **Hardware stub** raises `NotConfiguredError`; the prototype makes no network calls.

## Benchmarks and metrics
- **Reference edge map.** Synthetic scenes use their exact region boundaries as ground truth.
  For uploaded photos there is no ground truth, so Canny's output is the reference and Canny
  scores F1 = 1 by construction. The UI and README state this.
- **Ground-truth convention favours QHED.** The synthetic ground truth marks the pixel to the
  left of / above each boundary, which is exactly where a forward difference fires, so QHED
  reaches F1 = 1.0 there. Classical 3x3 operators produce 2-pixel-wide responses, which the
  1-pixel tolerance only partly forgives. Read the F1 numbers as "correct", not "better".
- **No quantum speedup.** Measured on the development machine (8 cores, NumPy fast path,
  thread pool): QHED simulation runs at about 4-7 MP/s, while OpenCV Sobel, Prewitt, Canny
  and Laplacian run at about 40-100 MP/s. At 4K that is 1.1 s for QHED versus 0.1-0.2 s
  classically. Both scale linearly with pixel count; the tiled design keeps the circuit
  size constant (9 qubits for 16x16 tiles) at any resolution.
- **Numerical noise floor.** `normalise` treats peaks below 1e-9 as zero, so rounding residue
  on flat images is not stretched into false full-scale edges (found by the classical tests).
- **Peak memory** comes from `tracemalloc`. It sees NumPy allocations but not native OpenCV or
  Aer buffers, so it is a lower bound.
