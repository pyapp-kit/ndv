from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Iterator

import dask.array as da
import numpy as np
import pytest
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent

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


@pytest.fixture(params=BACKENDS)
def viewer(
    qtbot: QtBot, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[NDViewer]:
    monkeypatch.setenv("NDV_CANVAS_BACKEND", request.param)
    v = NDViewer()
    qtbot.addWidget(v)
    yield v
    # wait until there are no running jobs, because the callbacks
    # in the futures hold a strong reference to the viewer
    qtbot.waitUntil(v._is_idle, timeout=3000)


@allow_linux_widget_leaks
def test_ndviewer(qtbot: QtBot, viewer: NDViewer) -> None:
    data = np.empty((4, 3, 2, 32, 32), dtype=np.uint8)
    viewer.set_data(data)
    viewer.show()
    qtbot.waitUntil(viewer._is_idle, timeout=1000)
    viewer.set_ndim(3)
    viewer.set_channel_mode("composite")
    viewer.set_current_index({0: 2, 1: 0, 2: 1})


@pytest.mark.allow_leaks
def test_hover_info(qtbot: QtBot) -> None:
    data = np.ones((4, 3, 32, 32), dtype=np.uint8)
    viewer = NDViewer(data)
    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitUntil(viewer._is_idle, timeout=1000)
    mouse_event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(200, 200),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewer.eventFilter(viewer._qcanvas, mouse_event)
    info_text = viewer._hover_info_label.text()
    assert info_text.endswith("0: 1")
