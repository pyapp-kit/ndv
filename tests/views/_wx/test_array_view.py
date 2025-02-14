from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from pytest import fixture

from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._wx._array_view import WxArrayView
from ndv.views.bases._array_view import ArrayViewOptions

if TYPE_CHECKING:
    import wx


@fixture
def viewer(wxapp: wx.App) -> WxArrayView:
    return WxArrayView(MagicMock(), _ArrayDataDisplayModel(), ArrayViewerModel())


def test_array_options(viewer: WxArrayView) -> None:
    wxwdg = viewer._wxwidget
    wxwdg.Show()
    assert wxwdg.ndims_btn.IsShown()
    # assert wxwdg.histogram_btn.IsShown()
    assert wxwdg.reset_zoom_btn.IsShown()
    assert wxwdg.channel_mode_combo.IsShown()
    assert wxwdg.add_roi_btn.IsShown()

    options = ArrayViewOptions(
        show_3d_button=False,
        show_channel_mode_selector=False,
        show_histogram_button=False,
        show_reset_zoom_button=False,
        show_roi_button=False,
    )
    viewer.set_options(options)

    assert not wxwdg.ndims_btn.IsShown()
    # assert not wxwdg.histogram_btn.IsShown()
    assert not wxwdg.reset_zoom_btn.IsShown()
    assert not wxwdg.channel_mode_combo.IsShown()
    assert not wxwdg.add_roi_btn.IsShown()
