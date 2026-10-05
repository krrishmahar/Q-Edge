"""Smoke tests: package imports and core dependencies are available."""

import importlib

import pytest

import q_edge


def test_version() -> None:
    assert q_edge.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module",
    [
        "q_edge.quantum",
        "q_edge.tiling",
        "q_edge.backends",
        "q_edge.scheduler",
        "q_edge.classical",
        "q_edge.metrics",
        "q_edge.config",
        "q_edge.pipeline",
        "qiskit",
        "qiskit_aer",
        "cv2",
        "skimage",
    ],
)
def test_imports(module: str) -> None:
    importlib.import_module(module)
