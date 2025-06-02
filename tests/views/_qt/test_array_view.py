from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from pytest import fixture
from qtpy.QtWidgets import QWidget

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._app import get_histogram_canvas_class
from ndv.views._qt._array_view import PlayButton, QtArrayView

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@fixture
def viewer(qtbot: QtBot) -> QtArrayView:
    viewer = QtArrayView(QWidget(), ArrayViewerModel())
    viewer.add_lut_view(None)
    viewer.create_sliders({0: range(10), 1: range(64), 2: range(128)})
    qtbot.addWidget(viewer.frontend_widget())
    return viewer


def test_array_options(viewer: QtArrayView) -> None:
    qwdg = viewer._qwidget
    qwdg.show()
    qlut = viewer._luts[None]._qwidget
    dims_wdg = viewer._qwidget.dims_sliders
    assert dims_wdg._sliders
    play_btn = dims_wdg._layout.itemAtPosition(1, dims_wdg._rPLAY_BTN).widget()  # type: ignore

    assert qwdg.ndims_btn.isVisible()
    viewer._viewer_model.show_3d_button = False
    assert not qwdg.ndims_btn.isVisible()

    assert qlut.histogram_btn.isVisible()
    viewer._viewer_model.show_histogram_button = False
    assert not qlut.histogram_btn.isVisible()

    assert qwdg.set_range_btn.isVisible()
    viewer._viewer_model.show_reset_zoom_button = False
    assert not qwdg.set_range_btn.isVisible()

    assert qwdg.channel_mode_combo.isVisible()
    viewer._viewer_model.show_channel_mode_selector = False
    assert not qwdg.channel_mode_combo.isVisible()

    assert qwdg.add_roi_btn.isVisible()
    viewer._viewer_model.show_roi_button = False
    assert not qwdg.add_roi_btn.isVisible()

    assert isinstance(play_btn, PlayButton)
    assert play_btn.isVisible()
    viewer._viewer_model.show_play_button = False
    assert not play_btn.isVisible()


def test_histogram(viewer: QtArrayView) -> None:
    channel = None
    lut = viewer._luts[channel]

    # Ensure lut signal gets passed through the viewer with the channel as the arg
    histogram_mock = Mock()
    viewer.histogramRequested.connect(histogram_mock)
    lut._qwidget.histogram_btn.setChecked(True)
    histogram_mock.assert_called_once_with(channel)

    # Test adding the histogram widget puts it on the relevant lut
    assert lut.histogram is None
    histogram = get_histogram_canvas_class()()  # will raise if not supported
    viewer.add_histogram(channel, histogram)
    assert lut.histogram is not None


def test_play_btn(viewer: QtArrayView, qtbot: QtBot) -> None:
    """Test the play button functionality on the array view."""
    dims_wdg = viewer._qwidget.dims_sliders
    assert dims_wdg._sliders
    play_btn = dims_wdg._layout.itemAtPosition(1, dims_wdg._rPLAY_BTN).widget()  # type: ignore
    assert isinstance(play_btn, PlayButton)
    play_btn._show_fps_dialog()
    play_btn._popup.accept()
    with qtbot.waitSignal(dims_wdg.currentIndexChanged, timeout=1000):
        play_btn.click()
    play_btn.click()  # stop it
