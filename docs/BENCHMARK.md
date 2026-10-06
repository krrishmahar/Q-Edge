# Q-Edge benchmark and validation

This report records measurements from the current Q-Edge implementation. It covers the
Hadamard-based Quantum Hadamard Edge Detection (QHED) algorithm running as a classical
statevector simulation through the NumPy CPU backend and the CuPy CUDA backend.

## Test environment

| Item | Value |
|---|---|
| Operating system | Windows |
| Python | 3.12.15 |
| NumPy | 2.5.3 |
| CuPy | 14.2.0 |
| CUDA toolkit | 13.4.2 |
| NVIDIA driver | 617.14 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Compute capability | 8.6 |
| GPU memory | 6143.5 MiB reported by CuPy |
| Revision | `0cf163b` |

The GPU was verified with a real CuPy kernel before benchmarking:

```text
Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CuPy result: [2, 3, 4]
```

## CPU versus CUDA QHED

The same synthetic input was run through `run_pipeline()` with `backend="numpy"` and
`backend="gpu"`. Both used one worker, the same 16x16/8x8 tile selection, and a full-resolution
input. Each row includes one warm-up run followed by three timed runs; the reported runtime is
the median of the timed runs.

| Resolution | Megapixels | Tiles | NumPy median (s) | NumPy MP/s | CUDA median (s) | CUDA MP/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512x512 | 0.2621 | 5,476 | 0.040782 | 6.428 | 0.010332 | 25.373 | 3.95x |
| 1280x720 | 0.9216 | 18,849 | 0.232143 | 3.970 | 0.050653 | 18.194 | 4.58x |
| 1920x1080 | 2.0736 | 42,625 | 0.310411 | 6.680 | 0.065478 | 31.669 | 4.74x |
| 3840x2160 | 8.2944 | 169,641 | 0.621703 | 13.341 | 0.236522 | 35.068 | 2.63x |

Speedup is `NumPy median / CUDA median`. These are measurements of classical simulation
acceleration; they are not quantum speedup.

## Image-quality validation

The 1920x1080 synthetic scene has exact forward-difference ground truth. The benchmark uses
the project’s one-pixel tolerance for binary edge metrics.

| Method | Runtime (s) | MP/s | SSIM | PSNR (dB) | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QHED / NumPy | 0.183322 | 11.311 | 0.9905 | 30.3389 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sobel | 0.043179 | 48.023 | 0.9847 | 26.3052 | 0.9857 | 1.0000 | 0.9928 | 0.4984 |
| Prewitt | 0.024046 | 86.234 | 0.9840 | 25.9248 | 0.9636 | 1.0000 | 0.9815 | 0.4639 |
| Canny | 0.024584 | 84.348 | 0.9944 | 30.3478 | 1.0000 | 0.9996 | 0.9998 | 0.7851 |
| Laplacian | 0.019112 | 108.500 | 0.9809 | 26.6355 | 0.9837 | 0.9653 | 0.9744 | 0.4692 |

The ground-truth convention places boundaries where QHED’s forward difference fires. Therefore,
QHED’s F1 of 1.0 confirms agreement with this ground truth; it does not establish superiority
over Canny or other detectors on general photographs.

## Numerical and circuit validation

For 128 random 16x16 tiles, the CUDA and NumPy QHED outputs were compared element by element:

| Check | Result |
|---|---:|
| Output shape | `(128, 15, 15)` |
| Maximum absolute error | `4.440892098500626e-16` |
| Mean absolute error | `3.4405153181481083e-17` |

The separate Qiskit Aer validation remains:

- 64x64 crop with 4x4 tiles
- Maximum fast-path versus Aer absolute difference: `5.76e-12`
- Aer execution time: approximately 12 seconds in the documented run

## Automated validation

The complete test suite passed:

```text
169 passed in 20.40s
```

The focused backend and UI suite passed after the CUDA packaging update:

```text
46 passed in 13.63s
```

The checks include backend agreement, CUDA capability reporting, JSON-safe UI metadata, tiling,
quality metrics, pipeline behavior, and Streamlit app startup.

## Reproduction

Install the CPU environment:

```powershell
uv sync
```

Install the CUDA extra:

```powershell
uv sync --extra gpu
```

Verify the device:

```powershell
uv run python -c "from q_edge.backends.gpu_backend import gpu_status; print(gpu_status())"
```

Run the project demo:

```powershell
uv run python scripts/demo.py --scaling
```

The CPU/GPU table in this report was generated with `run_pipeline()` using one worker and
three timed repetitions per resolution. On another machine, report its GPU, driver, CuPy,
CUDA, worker count, and repetition policy alongside the results.

## Limitations

- QHED is simulated classically. Neither the NumPy nor CUDA result demonstrates quantum
  advantage.
- GPU speed depends on image size, tile batch size, host-device transfer overhead, CUDA
  version, driver, and other processes using the device.
- The benchmark compares one CPU worker with one GPU worker. It is not a comparison against
  the maximum CPU thread-pool performance.
- CuPy’s GPU backend returns results to NumPy for the existing pipeline/stitching boundary;
  a future implementation could reduce transfers by keeping more stages on the device.
- Aer is a circuit simulator, not quantum hardware. Its timings include circuit simulation and
  transpilation costs.
- Amplitude encoding and full readout costs for real hardware are not modelled.
- Synthetic ground truth is favorable to a forward-difference detector.
- Uploaded photographs do not have labeled ground truth; the UI uses Canny as a reference.
- The IBM hardware backend remains a deliberate stub and makes no network calls.
- CUDA support is optional. When the capability probe cannot execute a real CuPy kernel, the
  UI disables the GPU option and selected GPU requests fail explicitly rather than silently
  running on the CPU.
