from __future__ import annotations

import sys
import unittest.mock
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

if all(x not in {"--codspeed", "tests/test_benchmarks.py"} for x in sys.argv):
    pytest.skip("use --benchmark to run benchmark", allow_module_level=True)


def _show_viewer(data: np.ndarray) -> None:
    import ndv

    ndv.imshow(data)


def test_time_to_show(benchmark: BenchmarkFixture, qapp: Any) -> None:
    data = np.random.randint(0, 255, size=(10, 256, 256), dtype=np.uint8)
    for k in list(sys.modules):
        if k.startswith(("ndv", "superqt", "vispy", "pygfx", "wgpu", "PyQt", "PySide")):
            del sys.modules[k]
    with unittest.mock.patch.object(qapp, "exec"):
        benchmark.pedantic(_show_viewer, (data,), iterations=1, rounds=1)
