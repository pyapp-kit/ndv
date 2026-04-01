from __future__ import annotations

from unittest.mock import MagicMock, Mock

import wx
from pytest import fixture

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._app import get_histogram_canvas_class
from ndv.views._wx._array_view import WxArrayView


@fixture
def viewer(wxapp: wx.App) -> WxArrayView:
    viewer = WxArrayView(MagicMock(), ArrayViewerModel())
    viewer.add_lut_view(None)
    return viewer


def _processEvent(
    wxapp: wx.App, evt: wx.PyEventBinder, wdg: wx.Control, **kwargs
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

    # Per-channel histogram buttons are hidden when use_shared_histogram=True
    assert not wxlut.histogram_btn.IsShown()
    viewer._viewer_model.use_shared_histogram = False
    assert wxlut.histogram_btn.IsShown()
    viewer._viewer_model.show_histogram_button = False
    assert not wxlut.histogram_btn.IsShown()

    assert wxwdg.set_range_btn.IsShown()
    viewer._viewer_model.show_reset_zoom_button = False
    assert not wxwdg.set_range_btn.IsShown()

    assert wxwdg.channel_mode_combo.IsShown()
    viewer._viewer_model.show_channel_mode_selector = False
    assert not wxwdg.channel_mode_combo.IsShown()

    assert not wxwdg.add_roi_btn.IsShown()
    viewer._viewer_model.show_roi_button = True
    assert wxwdg.add_roi_btn.IsShown()


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
    ch = 0
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 2):
        viewer.add_lut_view(ch)

    assert viewer._wxwidget.lut_selector.IsEnabled()

    assert not viewer._wxwidget.lut_selector.IsShown()
    assert not viewer._wxwidget._lut_toolbar_panel.IsShown()

    viewer.add_lut_view(ch + 1)

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

    checklist = viewer._wxwidget.lut_selector._checklist

    # display off for a single channel
    checklist.Check(0, False)
    _processEvent(wxapp, wx.EVT_CHECKLISTBOX, checklist)

    assert not checklist.IsChecked(0)
    assert not viewer._luts[0]._wxwidget.IsShown()

    # display on again for a single channel
    checklist.Check(0, True)
    _processEvent(wxapp, wx.EVT_CHECKLISTBOX, checklist)

    assert checklist.IsChecked(0)
    assert viewer._luts[0]._wxwidget.IsShown()

    # Simulate what the controller does on mode change to grayscale:
    # it calls set_visible(False) on all multi-channel LUTs
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int:
            lut_view.set_visible(False)
            assert not lut_view._wxwidget.IsShown()

    # Simulate switching back to composite: controller calls set_visible(True)
    # All previously-displayed channels should reappear
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int:
            lut_view.set_visible(True)
            assert lut_view._wxwidget.IsShown(), (
                f"Channel {ch} should be visible after set_visible(True)"
            )

    # Now test interaction with display selector:
    # If a channel was deselected via the channel selector (set_display(False)),
    # it should NOT reappear after a mode round-trip
    viewer._luts[0].set_display(False)
    assert not viewer._luts[0]._wxwidget.IsShown()

    # Hide all (mode change to grayscale)
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int:
            lut_view.set_visible(False)

    # Show all (mode change back to composite)
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int:
            lut_view.set_visible(True)

    # Channel 0 was display-hidden, so it should stay hidden
    assert not viewer._luts[0]._wxwidget.IsShown()
    # Other channels should be visible
    assert viewer._luts[1]._wxwidget.IsShown()


def test_removed_channels(wxapp: wx.App, viewer: WxArrayView) -> None:
    # display options button should appear only after thresh is reached
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        viewer.add_lut_view(ch)

    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        lut_view = viewer._luts[ch]
        viewer.remove_lut_view(lut_view)

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


def test_none_all(wxapp: wx.App, viewer: WxArrayView) -> None:
    for ch in range(viewer._wxwidget._toolbar_display_thresh - 1):
        viewer.add_lut_view(ch)

    none_btn = viewer._wxwidget.lut_selector._select_none_btn
    all_btn = viewer._wxwidget.lut_selector._select_all_btn

    # select none
    _processEvent(wxapp, wx.EVT_BUTTON, none_btn)

    # all channels should be hidden
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int or (type(ch) is str and ch.isdigit()):
            assert not lut_view._wxwidget.IsShown()

    # select all
    _processEvent(wxapp, wx.EVT_BUTTON, all_btn)

    # all channels should be displayed
    for ch, lut_view in viewer._luts.items():
        if type(ch) is int or (type(ch) is str and ch.isdigit()):
            assert lut_view._wxwidget.IsShown()


def test_key_event_filter(wxapp: wx.App) -> None:
    from ndv._types import KeyCode, KeyMod, KeyPressEvent
    from ndv.views._wx._app import WxAppWrap

    app = WxAppWrap()
    view = WxArrayView(MagicMock(), ArrayViewerModel())
    widget = view.frontend_widget()

    received: list[KeyPressEvent] = []
    view.keyPressed.connect(received.append)

    disconnect = app.filter_key_events(widget, view)

    # Simulate a Right arrow key press
    event = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    event.SetKeyCode(wx.WXK_RIGHT)
    wx.PostEvent(widget.GetEventHandler(), event)
    evtLoop = wxapp.GetTraits().CreateEventLoop()
    wx.EventLoopActivator(evtLoop)
    evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)

    assert len(received) == 1
    assert received[0].key == KeyCode.RIGHT
    assert received[0].mods == KeyMod.NONE

    disconnect()
