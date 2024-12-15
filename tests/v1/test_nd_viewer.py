from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

try:
    import pytestqt  # noqa: F401
except ImportError:
    pytest.skip("This module requires qt frontend", allow_module_level=True)


from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent

from ndv.old_viewer import NDViewer

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def allow_linux_widget_leaks(func: Any) -> Any:
    if sys.platform == "linux":
        pytest.mark.allow_leaks(func)
    return func


def test_empty_viewer(qtbot: QtBot) -> None:
    viewer = NDViewer()
    qtbot.add_widget(viewer)
    viewer.refresh()
    viewer.set_data(np.random.rand(4, 3, 32, 32))
    assert isinstance(viewer.data, np.ndarray)
    viewer.set_data(None)
    assert viewer.data is None


@allow_linux_widget_leaks
@pytest.mark.usefixtures("any_app")
def test_ndviewer() -> None:
    dask_arr = np.empty((4, 3, 2, 32, 32), dtype=np.uint8)
    v = NDViewer(dask_arr)
    # qtbot.addWidget(v)
    v.show()
    # qtbot.waitUntil(v._is_idle, timeout=1000)
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
def test_hover_info(qtbot: QtBot) -> None:
    data = np.ones((4, 3, 32, 32), dtype=np.float32)
    viewer = NDViewer(data)
    qtbot.addWidget(viewer)
    viewer.show()
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
