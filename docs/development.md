# Development guide

## Toolchain

| Tool | Use |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Environment, dependencies and running commands. Never use pip or `python -m venv` |
| ruff | Lint (`E, F, W, I, UP, B, SIM, NPY, RUF`) and formatting, line length 100 |
| mypy | Strict type checking of `src` and `app` |
| pytest + pytest-cov | Tests and coverage (the gate fails under 80%) |
| just | Task runner (`uv tool install rust-just`) |

```bash
uv sync                    # install from uv.lock
uv add <pkg>               # runtime dependency
uv add --dev <pkg>         # dev dependency
```

## Common tasks

| `just` recipe | Equivalent command |
|---|---|
| `just test` | `uv run pytest -q -m "not slow"` |
| `just test-all` | `uv run pytest -q` (includes the 4K end-to-end test) |
| `just lint` | `uv run ruff check . && uv run ruff format --check . && uv run mypy src app` |
| `just fmt` | `uv run ruff format . && uv run ruff check --fix .` |
| `just cov` | `uv run pytest -q --cov=q_edge --cov-report=term-missing` |
| `just app` | `uv run streamlit run app/streamlit_app.py` |
| `just demo` | `uv run python scripts/demo.py` |

## Test suite

| File | Covers |
|---|---|
| `tests/test_smoke.py` | Imports and version |
| `tests/test_quantum.py` | Encoding, the decrement unitary, gradient/constant/zero tiles, fast vs circuit agreement, shots, circuit size |
| `tests/test_tiling.py` | Lossless round trip, tiled equals untiled, seams, odd sizes, batches, validation |
| `tests/test_pipeline.py` | Config validation, loaders (bytes/path/PIL/array/16-bit), preprocess, pipeline, 4K (`slow`) |
| `tests/test_backends.py` | Registry, plug-in backend, backend agreement, GPU capability and pickling, hardware stub, parallel equals serial, resources |
| `tests/test_metrics.py` | Classical detectors, SSIM/PSNR/F1, tolerance, perf, `benchmark()`, `scaling_benchmark()` |
| `tests/test_ui.py` | Upload validation and hardening, helpers, charts, and headless `AppTest` runs of the Streamlit app |

Status at the last run: **165 passed**, coverage about **98%**, ruff and mypy clean.

## Validating the app headless

```bash
uv run streamlit run app/streamlit_app.py --server.headless true --server.port 8599
curl http://localhost:8599/_stcore/health     # -> ok
```

For CUDA validation, install the optional dependency and query the real device:

```bash
uv sync --extra gpu
uv run python -c "from q_edge.backends.gpu_backend import gpu_status; print(gpu_status())"
```

The status is usable only after a CuPy kernel succeeds. The embedded frontend receives the
same capability data from `/api/capabilities` and disables the GPU option when it is not usable.

`AppTest` in `tests/test_ui.py` executes the full script (sample image, Aer backend, hardware
stub) without a browser.

## Regenerating screenshots

Playwright is deliberately **not** a project dependency. Run it ad hoc against a local Chrome:

```bash
uv run streamlit run app/streamlit_app.py --server.headless true --server.port 8601
uv run --with playwright python scripts/capture_screenshots.py --url http://localhost:8601
```

The script drives the app through eight states and writes `docs/screenshots/01-...08-*.png`.

## Conventions

- Every public function and class is typed and has a docstring.
- Backends must be picklable, so the process executor can ship them to workers.
- No runtime network calls and no secrets in the repository.
- Each piece of work happens on its own `wave/N-name` branch and is squash-merged into `main`
  with a conventional commit message.
- Ambiguous requirements get a sensible default, recorded in [DECISIONS.md](../DECISIONS.md).

## Project history

| Commit | Wave |
|---|---|
| `chore: scaffold q-edge project` | 0: uv, tooling, layout |
| `feat(quantum): ...` | 1: encoding, fast and circuit QHED |
| `feat(tiling): ...` | 2: halo tiler, config, pipeline, 4K run |
| `feat(backends): ...` | 3: registry, four backends, scheduler |
| `feat(metrics): ...` | 4: classical baselines, metrics, benchmarks |
| `feat(app): ...` | 5: Streamlit UI |
| `docs: harden uploads, add demo script and README` | 6: hardening and docs |
| `docs: add docs folder with screenshots ...` | 7: this documentation set |
