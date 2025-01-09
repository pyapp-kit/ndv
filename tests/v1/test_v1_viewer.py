from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

try:
    import pytestqt

    if pytestqt.qt_compat.qt_api.pytest_qt_api.startswith("pyside"):
        pytest.skip("V1 viewer segfaults with pyside", allow_module_level=True)

except ImportError:
    pytest.skip("This module requires qt frontend", allow_module_level=True)


from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent

from ndv.v1 import NDViewer


def allow_linux_widget_leaks(func: Any) -> Any:
    if sys.platform == "linux":
        pytest.mark.allow_leaks(func)
    return func


@allow_linux_widget_leaks
@pytest.mark.usefixtures("any_app")
def test_empty_viewer() -> None:
    viewer = NDViewer()
    viewer.refresh()
    viewer.set_data(np.random.rand(4, 3, 32, 32))
    assert isinstance(viewer.data, np.ndarray)
    viewer.set_data(None)
    assert viewer.data is None


@allow_linux_widget_leaks
def test_ndviewer(any_app: Any) -> None:
    dask_arr = np.empty((4, 3, 2, 32, 32), dtype=np.uint8)
    v = NDViewer(dask_arr)
    # qtbot.addWidget(v)
    v.show()
    if isinstance(any_app, tuple) and len(any_app) == 2:
        qtbot = any_app[1]
        qtbot.waitUntil(v._is_idle, timeout=1000)
    v.set_ndim(3)
    v.set_channel_mode("composite")
    v.set_current_index({0: 2, 1: 1, 2: 1})
    v.set_roi(
        [(10, 10), (30, 10), (30, 30), (10, 30)],
        color="blue",
        border_color="light blue",
    )

    # wait until there are no running jobs, because the callbacks
    # in the futures hold a strong reference to the viewer
    # qtbot.waitUntil(v._is_idle, timeout=3000)


# not testing pygfx yet...
@pytest.mark.skipif(sys.platform != "darwin", reason="the mouse event is tricky")
def test_hover_info(any_app: Any) -> None:
    data = np.ones((4, 3, 32, 32), dtype=np.float32)
    viewer = NDViewer(data)
    viewer.show()
    if isinstance(any_app, tuple) and len(any_app) == 2:
        qtbot = any_app[1]
        qtbot.waitUntil(viewer._is_idle, timeout=1000)
    mouse_event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(viewer._qcanvas.rect().center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewer.eventFilter(viewer._qcanvas, mouse_event)
    info_text = viewer._hover_info_label.text()
    assert info_text.endswith("0: 1.00")
