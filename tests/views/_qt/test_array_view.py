from __future__ import annotations

from typing import TYPE_CHECKING

from pytest import fixture
from qtpy.QtWidgets import QWidget

from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._qt._array_view import QtArrayView

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@fixture
def viewer(qtbot: QtBot) -> QtArrayView:
    viewer = QtArrayView(QWidget(), _ArrayDataDisplayModel(), ArrayViewerModel())
    viewer.add_lut_view(None)
    qtbot.addWidget(viewer.frontend_widget())
    return viewer


def test_array_options(viewer: QtArrayView) -> None:
    qwdg = viewer._qwidget
    qwdg.show()
    qlut = viewer._luts[None]._qwidget

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
