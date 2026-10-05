"""Tests for execution backends, the registry, resources and the parallel executor."""

import pickle
from dataclasses import dataclass

import numpy as np
import pytest

from q_edge.backends import (
    AerBackend,
    Backend,
    GpuBackend,
    IBMHardwareBackend,
    NotConfiguredError,
    NumpyBackend,
    available_backends,
    get_backend,
    register_backend,
)
from q_edge.backends.gpu_backend import qhed_edge_magnitude_xp
from q_edge.config import ExecutorKind, QEdgeConfig
from q_edge.pipeline import run_pipeline
from q_edge.quantum.qhed_fast import qhed_edge_magnitude
from q_edge.scheduler.executor import execute
from q_edge.scheduler.resources import (
    SystemResources,
    auto_settings,
    detect_memory,
    detect_resources,
)
from q_edge.tiling.tiler import split

SMALL = np.random.default_rng(42).random((12, 14))


# --- registry -----------------------------------------------------------------------------


def test_builtin_backends_registered() -> None:
    assert {"numpy", "aer", "gpu", "ibm-hardware"} <= set(available_backends())
    for name in available_backends():
        assert isinstance(get_backend(name), Backend)


def test_unknown_backend_lists_alternatives() -> None:
    with pytest.raises(ValueError, match="available: "):
        get_backend("does-not-exist")


def test_duplicate_registration_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_backend("numpy", NumpyBackend.from_config)


def test_aer_factory_reads_config() -> None:
    backend = get_backend("aer", QEdgeConfig(backend="aer", shots=123, seed=5))
    assert isinstance(backend, AerBackend)
    assert (backend.shots, backend.seed) == (123, 5)


@dataclass(frozen=True)
class _DoublingBackend:
    """A custom backend added without touching the pipeline."""

    name: str = "doubling"

    def run_tiles(self, batch: np.ndarray) -> np.ndarray:
        return 2.0 * qhed_edge_magnitude(batch)


def test_custom_backend_plugs_into_pipeline() -> None:
    register_backend("doubling", lambda cfg: _DoublingBackend(), overwrite=True)
    cfg = QEdgeConfig(backend="doubling", tile_size=4)
    custom = run_pipeline(SMALL, cfg)
    reference = run_pipeline(SMALL, cfg.with_updates(backend="numpy"))
    np.testing.assert_allclose(custom.magnitude, 2.0 * reference.magnitude)


# --- backend agreement --------------------------------------------------------------------


@pytest.mark.parametrize("tile_size", [4, 8])
def test_all_backends_agree_on_small_image(tile_size: int) -> None:
    base = QEdgeConfig(tile_size=tile_size, workers=1)
    reference = run_pipeline(SMALL, base).magnitude
    for backend in (NumpyBackend(), AerBackend(), GpuBackend()):
        result = run_pipeline(SMALL, base, backend=backend).magnitude
        np.testing.assert_allclose(result, reference, atol=1e-6, err_msg=backend.name)


def test_aer_shot_backend_close_to_exact() -> None:
    cfg = QEdgeConfig(tile_size=4, workers=1)
    exact = run_pipeline(SMALL, cfg).magnitude
    sampled = run_pipeline(SMALL, cfg, backend=AerBackend(shots=100_000, seed=1)).magnitude
    np.testing.assert_allclose(sampled, exact, atol=0.06)


def test_gpu_backend_falls_back_to_numpy() -> None:
    backend = GpuBackend()
    tiles = split(SMALL, 8).all_tiles()
    np.testing.assert_allclose(backend.run_tiles(tiles), qhed_edge_magnitude(tiles))
    if not backend.uses_gpu:
        assert backend.run_tiles(tiles).dtype == np.float64


def test_gpu_backend_pickles_without_module_state() -> None:
    backend = GpuBackend()
    backend.run_tiles(split(SMALL, 4).all_tiles())
    clone = pickle.loads(pickle.dumps(backend))
    assert clone.name == "gpu"
    np.testing.assert_allclose(clone.run_tiles(np.ones((1, 4, 4))), 0.0)


def test_xp_implementation_matches_fast_path() -> None:
    tiles = np.random.default_rng(0).random((50, 8, 8))
    tiles[3] = 0.0
    np.testing.assert_allclose(
        qhed_edge_magnitude_xp(tiles, np), qhed_edge_magnitude(tiles), atol=1e-12
    )


def test_hardware_stub_fails_gracefully() -> None:
    backend = get_backend("ibm-hardware", QEdgeConfig(shots=1000))
    assert isinstance(backend, IBMHardwareBackend)
    assert backend.shots == 1000
    with pytest.raises(NotConfiguredError, match="not configured"):
        backend.run_tiles(np.ones((1, 4, 4)))
    with pytest.raises(NotConfiguredError):
        run_pipeline(SMALL, QEdgeConfig(backend="ibm-hardware"))


# --- executor -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [ExecutorKind.THREAD, ExecutorKind.PROCESS])
def test_parallel_output_equals_serial(kind: ExecutorKind) -> None:
    image = np.random.default_rng(3).random((150, 170))
    grid = split(image, 8)
    serial = execute(grid, NumpyBackend(), workers=1, batch_size=37)
    progress: list[int] = []
    parallel = execute(
        grid,
        NumpyBackend(),
        workers=3,
        batch_size=37,
        kind=kind,
        progress=lambda done, total: progress.append(done),
    )
    np.testing.assert_array_equal(parallel, serial)
    assert progress == sorted(progress) and progress[-1] == grid.num_tiles


def test_pipeline_parallel_equals_serial() -> None:
    image = np.random.default_rng(4).random((200, 260))
    cfg = QEdgeConfig(tile_size=16, batch_size=20)
    serial = run_pipeline(image, cfg.with_updates(executor="serial"))
    threaded = run_pipeline(image, cfg.with_updates(workers=4))
    np.testing.assert_array_equal(threaded.magnitude, serial.magnitude)
    assert threaded.workers == 4 and threaded.batch_size == 20


def test_executor_rejects_bad_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        execute(split(SMALL, 4), NumpyBackend(), workers=0)


# --- resources ----------------------------------------------------------------------------


def test_detect_resources_is_sane() -> None:
    res = detect_resources()
    assert res.cpu_count >= 1
    assert 0 < res.available_memory <= res.total_memory
    total, available = detect_memory()
    assert total > 0 and available > 0


_RES = SystemResources(
    cpu_count=8, total_memory=16 * 1024**3, available_memory=8 * 1024**3, gpu_available=False
)


def test_auto_settings_for_4k_simulation() -> None:
    s = auto_settings((2160, 3840), "numpy", _RES)
    assert s.tile_size == 16
    assert 256 <= s.batch_size <= 65_536
    assert 1 <= s.workers <= 8


def test_auto_settings_for_small_and_circuit_backends() -> None:
    small = auto_settings((32, 32), "numpy", _RES)
    assert small.tile_size == 8 and small.workers == 1
    circuit = auto_settings((256, 256), "aer", _RES)
    assert circuit.tile_size == 4 and circuit.batch_size == 32 and circuit.workers <= 4
