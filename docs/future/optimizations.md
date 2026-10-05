# Optimisation opportunities

Concrete improvements identified while building and measuring the prototype. Each entry names
the code involved, the expected effect and the risk.

Baseline (development machine, 8 cores, NumPy backend, threads): a 4K frame takes about 1.1 s
with 8x8 tiles and 0.9 s with 16x16 tiles. A 64x64 image on Aer takes about 13 s.

## Fast path (NumPy)

| # | Optimisation | Where | Expected effect | Risk |
|---|---|---|---|---|
| 1 | **float32 mode.** Simulate in single precision | `qhed_fast.py`, `tiler.py` | About 2x less memory and 1.3-1.8x faster | Differences around 1e-7; the fast vs circuit tests need a looser tolerance in that mode |
| 2 | **Skip the doubled statevector.** Compute odd amplitudes directly as `(a_i - a_{i+1})/2`, without materialising `(B, 2N)` | `qhed_differences` | About 3x less temporary memory, about 2x faster | Less visibly "statevector"; keep the full version as the reference and test both |
| 3 | **Fused kernel.** Numba or a C kernel for normalise → shift → difference → hypot in one pass | `qhed_fast.py` | 3-5x faster | Adds a dependency |
| 4 | **Contiguous batch slicing.** Replace the fancy-index copy in `TileGrid.batch` with whole-row slices | `tiler.py` | Removes one copy per batch | Batches become row-aligned |
| 5 | **Larger default tiles at 4K.** Halo overhead is `T^2/(T-1)^2`: 1.31x for 8x8, 1.14x for 16x16 | `auto_settings` | About 15% less work | Larger circuits on hardware (9 qubits) |
| 6 | **Robust normalisation.** Use the 99.5th percentile instead of the maximum | `imaging.normalise` | Better contrast when one strong edge dominates | Changes threshold semantics |

## Scheduling and parallelism

| # | Optimisation | Where | Expected effect | Risk |
|---|---|---|---|---|
| 7 | **Shared-memory process pool.** Put the padded image in `multiprocessing.shared_memory` and send only index ranges | `executor.py` | Removes pickling, which is why processes (3.2 s) lose to threads (1.1 s) at 4K | Lifecycle cleanup on Windows |
| 8 | **Measured auto-tuning.** Time a few batch sizes on first run and cache the best | `resources.py` | 10-30% on unusual hardware | First run is slower |
| 9 | **Use the batch-size budget.** Available RAM was low on the dev box, which gave batches of about 1,200 tiles; raise the fraction when RAM is plentiful | `_BATCH_MEMORY_FRACTION` | Fewer task overheads | Higher peak memory |

## Circuit path (Aer and hardware)

| # | Optimisation | Where | Expected effect | Risk |
|---|---|---|---|---|
| 10 | **Transpile once.** Build a parameterised template per tile size and bind amplitudes per tile | `qhed_circuit.py` | Large share of the 13 s Aer time is transpilation; likely 5-10x faster | State-prep parameterisation is non-trivial |
| 11 | **Run both directions in one job.** Submit horizontal and vertical circuits together | `circuit_differences`, `oriented_responses` | Halves the number of `run` calls | Small refactor |
| 12 | **Aer parallel experiments.** Set `max_parallel_experiments` and `max_parallel_threads` | `_run` | Multi-core use inside one call | Contention with the thread pool; set workers to 1 for Aer |
| 13 | **Direct Statevector for tiny circuits.** `qiskit.quantum_info.Statevector` avoids simulator overhead for 5-qubit circuits | `qhed_circuit.py` | Faster validation path | No longer "on Aer" |
| 14 | **Shot budgeting.** Allocate shots by tile variance and skip flat tiles (zero norm or zero variance) | Aer/hardware backends | Fewer shots for the same quality | Possible bias near weak edges |

## Streamlit app

| # | Optimisation | Where | Expected effect | Risk |
|---|---|---|---|---|
| 15 | **Reuse the QHED result in the metrics.** `run_benchmark` currently re-runs QHED twice (via `compute_quantum` and inside `benchmark`) | `app/streamlit_app.py`, `benchmark.py` | Removes two of the three QHED runs on first load | `benchmark` needs an optional precomputed result |
| 16 | **Cache classical maps.** Classical detectors re-run on every interaction | `results_tab` | Faster method switching at 4K | Cache memory |
| 17 | **Display `-` for not-applicable values.** `tiles_per_s` shows `None` for classical rows | `results_tab` formatting | Cosmetic fix | None |
| 18 | **Background scaling benchmark.** Run the sweep in a thread with a progress bar | `benchmarks_tab` | UI stays responsive | Streamlit threading caveats |
| 19 | **Image-level progress for Aer.** Report progress per circuit, not per batch | `AerBackend` | Smoother feedback on slow runs | None |

## Quality

| # | Optimisation | Expected effect |
|---|---|---|
| 20 | Gaussian pre-smoothing option before QHED, to reduce noise sensitivity on photos | Higher precision on real images |
| 21 | Non-maximum suppression and hysteresis on QHED magnitudes (a "quantum Canny") | Cleaner, connected edges |
| 22 | Central-difference variant via a second shifted copy | Symmetric, sub-pixel-centred edges |
| 23 | Real-photo ground truth (BSDS500) for the benchmark, instead of Canny as reference | Fairer F1 comparison |

## Suggested order

1. **#15 and #17:** quick UI wins.
2. **#2 and #1:** largest fast-path gains for little code.
3. **#10 and #11:** make Aer practical for larger demos.
4. **#7:** unlock process and distributed scaling (roadmap phase 4).
5. **#23:** credible quality claims on real images.
