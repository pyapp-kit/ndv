from __future__ import annotations

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


def _processEvent(
    wxapp: wx.App,
    evt: wx.PyEventBinder,
    wdg: wx.Control,
    **kwargs
) -> None:
    if evt == wx.EVT_ACTIVATE:
        active = kwargs.get("active", True)
        ev = wx.ActivateEvent(eventType=evt.typeId, active=active)
    else:
        ev = wx.PyCommandEvent(evt.typeId, wdg.GetId())

    wx.PostEvent(wdg.GetEventHandler(), ev)
    # Borrowed from:
    # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
    evtLoop = wxapp.GetTraits().CreateEventLoop()
    wx.EventLoopActivator(evtLoop)
    evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)


def test_array_options(viewer: WxArrayView) -> None:
    wxwdg = viewer._wxwidget
    wxwdg.Show()
    wxlut = viewer._luts[None]._wxwidget

    assert wxwdg.ndims_btn.IsShown()
    viewer._viewer_model.show_3d_button = False
    assert not wxwdg.ndims_btn.IsShown()

    assert wxlut.histogram_btn.IsShown()
    viewer._viewer_model.show_histogram_button = False
    assert not wxlut.histogram_btn.IsShown()

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
    channel = None
    lut = viewer._luts[channel]
    btn = lut._wxwidget.histogram_btn

    # Ensure lut signal gets passed through the viewer with the channel as the arg
    histogram_mock = Mock()
    viewer.histogramRequested.connect(histogram_mock)
    btn.SetValue(True)
    _processEvent(wxapp, wx.EVT_TOGGLEBUTTON, btn)
    histogram_mock.assert_called_once_with(channel)

    # Test adding the histogram widget puts it on the relevant lut
    assert len(lut._wxwidget._histogram_sizer.GetChildren()) == 1
    histogram = get_histogram_canvas_class()()  # will raise if not supported
    viewer.add_histogram(channel, histogram)
    assert len(lut._wxwidget._histogram_sizer.GetChildren()) == 2


# == Tests for display of channels ==


def test_display_options_visibility(wxapp: wx.App, viewer: WxArrayView) -> None:
    # display options button should appear only after thresh is reached
    # -2 to account for add_lut_view(None) in fixture
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 2):
        viewer.add_lut_view(ch)

    assert viewer._wxwidget.lut_selector.IsEnabled()

    assert not viewer._wxwidget._lut_toolbar_shown
    assert not viewer._wxwidget.lut_selector.IsShown()
    assert not viewer._wxwidget._lut_toolbar_panel.IsShown()

    viewer.add_lut_view(ch + 1)

    assert viewer._wxwidget._lut_toolbar_shown
    assert viewer._wxwidget.lut_selector.IsShown()
    assert viewer._wxwidget._lut_toolbar_panel.IsShown()


def test_display_options_selection(wxapp: wx.App, viewer: WxArrayView) -> None:
    # display options button should appear after thresh reached
    num_channels = viewer._wxwidget._toolbar_display_thresh - 1
    for ch in range(num_channels):
        viewer.add_lut_view(ch)

    assert len(viewer._wxwidget.luts.Children) == len(viewer._luts)

    # all channels should initially be displayed
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int or (type(ch) is str and ch.isdigit()):
            assert lut_view._wxwidget.IsShown()

    # display off for a single channel
    checklist = viewer._wxwidget.lut_selector._checklist
    checklist.Check(0, False)
    _processEvent(wxapp, wx.EVT_CHECKLISTBOX, checklist)

    assert not checklist.IsChecked(0)
    assert not viewer._luts[0]._wxwidget.IsShown()

    # channel_mode = viewer._wxwidget.channel_mode_combo
    # viewer.set_channel_mode(ChannelMode.GRAYSCALE)
    # _processEvent(wxapp, wx.EVT_COMBOBOX, channel_mode)

    ## all channels should be hidden
    # for ch, lut_view in viewer._luts.items():
    #    if type(ch) is int or (type(ch) is str and ch.isdigit()):
    #        assert not lut_view._wxwidget.IsShown()

def test_removed_channels(wxapp: wx.App, viewer: WxArrayView) -> None:
    # display options button should appear only after thresh is reached
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        viewer.add_lut_view(ch)

    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        lut_view = viewer._luts[ch]
        viewer.remove_lut_view(lut_view)

    assert not viewer._wxwidget._lut_toolbar_shown
    assert not viewer._wxwidget.lut_selector.IsShown()
    assert not viewer._wxwidget._lut_toolbar_panel.IsShown()

    # len == 1 to account for the None key
    assert len(viewer._luts) == 1
    assert len(viewer._wxwidget.luts.Children) == 1
    assert len(viewer._wxwidget.lut_selector._checklist.Children) == 0

def test_dropdown_popup(wxapp: wx.App, viewer: WxArrayView) -> None:
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        viewer.add_lut_view(ch)

    assert not viewer._wxwidget.lut_selector._popup.IsShown()

    ch_selection_dropdown = viewer._wxwidget.lut_selector._dropdown_btn
    _processEvent(wxapp, wx.EVT_BUTTON, ch_selection_dropdown)

    assert viewer._wxwidget.lut_selector._popup.IsShown()

    _processEvent(
        wxapp, wx.EVT_ACTIVATE, viewer._wxwidget.lut_selector._popup, active=False
    )

    assert not viewer._wxwidget.lut_selector._popup.IsShown()
