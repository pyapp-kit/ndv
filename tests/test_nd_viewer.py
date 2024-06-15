from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pytest

from ndv import NDViewer

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def allow_linux_widget_leaks(func: Any) -> Any:
    if sys.platform == "linux":
        pytest.mark.allow_leaks(func)
    return func


def make_lazy_array(shape: tuple[int, ...]) -> da.Array:
    rest_shape = shape[:-2]
    frame_shape = shape[-2:]

    def _dask_block(block_id: tuple[int, int, int, int, int]) -> np.ndarray | None:
        if isinstance(block_id, np.ndarray):
            return None
        size = (1,) * len(rest_shape) + frame_shape
        return np.random.randint(0, 255, size=size, dtype=np.uint8)

    chunks = [(1,) * x for x in rest_shape] + [(x,) for x in frame_shape]
    return da.map_blocks(_dask_block, chunks=chunks, dtype=np.uint8)  # type: ignore


BACKENDS = ["vispy"]
# avoid pygfx backend on linux CI
if not os.getenv("CI") or sys.platform == "darwin":
    BACKENDS.append("pygfx")


@allow_linux_widget_leaks
@pytest.mark.filterwarnings("ignore:This version of pygfx does not yet")
@pytest.mark.parametrize("backend", BACKENDS)
def test_ndviewer(qtbot: QtBot, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NDV_CANVAS_BACKEND", backend)
    dask_arr = make_lazy_array((1000, 64, 3, 256, 256))
    v = NDViewer(dask_arr)
    qtbot.addWidget(v)
    v.show()
    qtbot.waitUntil(v._is_idle, timeout=1000)
    v.set_ndim(3)
    v.set_channel_mode("composite")
    v.set_current_index({0: 100, 1: 10, 2: 1})

    # wait until there are no running jobs, because the callbacks
    # in the futures hold a strong reference to the viewer
    qtbot.waitUntil(v._is_idle, timeout=3000)
