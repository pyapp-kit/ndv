from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, Mock

import wx
from pytest import fixture

from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._app import get_histogram_canvas_class
from ndv.views._wx._array_view import WxArrayView


@fixture
def viewer(wxapp: wx.App) -> WxArrayView:
    viewer = WxArrayView(MagicMock(), _ArrayDataDisplayModel(), ArrayViewerModel())
    viewer.add_lut_view(None)
    return viewer


def test_array_options(viewer: WxArrayView) -> None:
    wxwdg = viewer._wxwidget
    wxwdg.Show()
    wxlut = viewer._luts[None]._wxwidget

    assert wxwdg.ndims_btn.IsShown()
    viewer._viewer_model.show_3d_button = False
    assert not wxwdg.ndims_btn.IsShown()

    assert wxlut.histogram.IsShown()
    viewer._viewer_model.show_histogram_button = False
    assert not wxlut.histogram.IsShown()

    assert wxwdg.set_range_btn.IsShown()
    viewer._viewer_model.show_reset_zoom_button = False
    assert not wxwdg.set_range_btn.IsShown()

    assert wxwdg.channel_mode_combo.IsShown()
    viewer._viewer_model.show_channel_mode_selector = False
    assert not wxwdg.channel_mode_combo.IsShown()

    assert wxwdg.add_roi_btn.IsShown()
    viewer._viewer_model.show_roi_button = False
    assert not wxwdg.add_roi_btn.IsShown()


def test_histogram(wxapp: wx.App, viewer: WxArrayView) -> None:
    def processEvent(evt: wx.PyEventBinder, wdg: wx.Control) -> None:
        ev = wx.PyCommandEvent(evt.typeId, wdg.GetId())
        wx.PostEvent(wdg.GetEventHandler(), ev)
        # Borrowed from:
        # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
        evtLoop = wxapp.GetTraits().CreateEventLoop()
        wx.EventLoopActivator(evtLoop)
        evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)

    channel = None
    lut = viewer._luts[channel]
    btn = lut._wxwidget.histogram

    # Ensure lut signal gets passed through the viewer with the channel as the arg
    histogram_mock = Mock()
    viewer.histogramRequested.connect(histogram_mock)
    btn.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, btn)
    histogram_mock.assert_called_once_with(channel)

    # Test adding the histogram widget puts it on the relevant lut
    assert len(lut._wxwidget.sizer.GetChildren()) == 1
    histogram = get_histogram_canvas_class()()  # will raise if not supported
    histogram_wdg = cast("wx.Window", histogram.frontend_widget())
    viewer.add_histogram(channel, histogram_wdg)
    assert len(lut._wxwidget.sizer.GetChildren()) == 2
