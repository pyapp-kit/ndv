from __future__ import annotations

from unittest.mock import MagicMock

import cmap
import wx
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsMinMax, ClimsPercentile, LUTModel
from ndv.views._wx._array_view import WxLutView
from ndv.views.bases._graphics._canvas import HistogramCanvas


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(wxapp: wx.App, model: LUTModel) -> WxLutView:
    # NB: wx.App necessary although unused
    frame = wx.Frame(None)
    view = WxLutView(frame)
    assert view.model is None
    view.model = model
    assert view.model is model
    return view


def test_WxLutView_update_model(model: LUTModel, view: WxLutView) -> None:
    """Ensures the view updates when the model is changed."""

    # Test modifying model.clims
    assert view._wxwidget.auto_clim.GetValue()
    model.clims = ClimsManual(min=0, max=1)
    assert not view._wxwidget.auto_clim.GetValue()
    model.clims = ClimsPercentile(min_percentile=0, max_percentile=100)
    assert view._wxwidget.auto_clim.GetValue()
    model.clims = ClimsPercentile(min_percentile=1, max_percentile=99)
    assert view._wxwidget.lower_tail.GetValue() == 1
    assert view._wxwidget.upper_tail.GetValue() == 1

    # Test modifying model.visible
    assert view._wxwidget.visible.GetValue()
    model.visible = False
    assert not view._wxwidget.visible.GetValue()
    model.visible = True
    assert view._wxwidget.visible.GetValue()

    # Test modifying model.cmap
    new_cmap = cmap.Colormap("red")
    assert view._wxwidget.cmap.GetValue() != new_cmap
    model.cmap = new_cmap
    assert view._wxwidget.cmap.GetValue() == new_cmap


def test_WxLutView_update_view(wxapp: wx.App, model: LUTModel, view: WxLutView) -> None:
    """Ensures the model updates when the view is changed."""

    def processEvent(evt: wx.PyEventBinder, wdg: wx.Control) -> None:
        ev = wx.PyCommandEvent(evt.typeId, wdg.GetId())
        wx.PostEvent(wdg.GetEventHandler(), ev)
        # Borrowed from:
        # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
        evtLoop = wxapp.GetTraits().CreateEventLoop()
        wx.EventLoopActivator(evtLoop)
        evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)

    new_clims = (5, 6)
    assert model.clims != new_clims
    clim_wdg = view._wxwidget.clims
    clim_wdg.SetValue(*new_clims)
    processEvent(wx.EVT_SLIDER, clim_wdg)
    assert model.clims == ClimsManual(min=5, max=6)

    new_visible = not model.visible
    vis_wdg = view._wxwidget.visible
    vis_wdg.SetValue(new_visible)
    processEvent(wx.EVT_CHECKBOX, vis_wdg)
    assert model.visible == new_visible

    new_cmap = cmap.Colormap("red")
    assert model.cmap != new_cmap
    cmap_wdg = view._wxwidget.cmap
    cmap_wdg.SetValue(new_cmap.name)
    processEvent(wx.EVT_COMBOBOX, cmap_wdg)
    assert model.cmap == new_cmap

    # Test toggling auto_clim
    auto_wdg = view._wxwidget.auto_clim
    auto_wdg.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, auto_wdg)
    assert model.clims == ClimsPercentile(min_percentile=0, max_percentile=100)
    auto_wdg.SetValue(False)
    processEvent(wx.EVT_TOGGLEBUTTON, auto_wdg)
    mi, ma = view._wxwidget.clims.GetValues()
    assert model.clims == ClimsManual(min=mi, max=ma)

    # Test modifying tails changes percentiles
    auto_wdg.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, auto_wdg)
    assert model.clims == ClimsPercentile(min_percentile=0, max_percentile=100)
    lower_wdg = view._wxwidget.lower_tail
    lower_wdg.SetValue(0.1)
    processEvent(wx.EVT_SPINCTRLDOUBLE, lower_wdg)
    assert model.clims == ClimsPercentile(min_percentile=0.1, max_percentile=100)
    upper_wdg = view._wxwidget.upper_tail
    upper_wdg.SetValue(0.2)
    processEvent(wx.EVT_SPINCTRLDOUBLE, upper_wdg)
    assert model.clims == ClimsPercentile(min_percentile=0.1, max_percentile=99.8)

    # When gui clims change, autoscale should be disabled
    model.clims = ClimsMinMax()
    clim_wdg = view._wxwidget.clims
    clim_wdg.SetValue(0, 1)
    processEvent(wx.EVT_SLIDER, clim_wdg)
    assert model.clims == ClimsManual(min=0, max=1)


def test_WxLutView_histogram_controls(wxapp: wx.App, view: WxLutView) -> None:
    def processEvent(evt: wx.PyEventBinder, wdg: wx.Control) -> None:
        ev = wx.PyCommandEvent(evt.typeId, wdg.GetId())
        wx.PostEvent(wdg.GetEventHandler(), ev)
        # Borrowed from:
        # https://github.com/wxWidgets/Phoenix/blob/master/unittests/wtc.py#L41
        evtLoop = wxapp.GetTraits().CreateEventLoop()
        wx.EventLoopActivator(evtLoop)
        evtLoop.YieldFor(wx.EVT_CATEGORY_ALL)

    # Mock up a histogram
    hist_mock = MagicMock(spec=HistogramCanvas)
    # Note that containing the frontend widget within a frame prevents segfaults
    frame = wx.Frame(None)
    hist_frontend = wx.Window(frame)
    hist_mock.frontend_widget.return_value = hist_frontend

    # Add the histogram and assert it was correctly added
    view._add_histogram(hist_mock)
    assert view.histogram is hist_mock

    # Assert histogram button toggles visibility
    hist_btn = view._wxwidget.histogram_btn
    log_wdg = view._wxwidget.log_btn
    reset_wdg = view._wxwidget.set_hist_range_btn

    hist_btn.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, hist_btn)
    assert log_wdg.IsShown()
    assert reset_wdg.IsShown()
    assert hist_frontend.IsShown()

    hist_btn.SetValue(False)
    processEvent(wx.EVT_TOGGLEBUTTON, hist_btn)
    assert not log_wdg.IsShown()
    assert not reset_wdg.IsShown()
    assert not hist_frontend.IsShown()

    # Assert toggling the log button alters the logarithmic base
    log_wdg.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, log_wdg)
    hist_mock.set_log_base.assert_called_once_with(10)
    hist_mock.reset_mock()

    log_wdg.SetValue(False)
    processEvent(wx.EVT_TOGGLEBUTTON, log_wdg)
    hist_mock.set_log_base.assert_called_once_with(None)
    hist_mock.reset_mock()

    # Assert pressing the reset view button sets the histogram range
    processEvent(wx.EVT_BUTTON, reset_wdg)
    hist_mock.set_range.assert_called_once_with()
    hist_mock.reset_mock()

    # Assert pressing the reset view button turns off log mode
    log_wdg.SetValue(True)
    processEvent(wx.EVT_TOGGLEBUTTON, log_wdg)
    processEvent(wx.EVT_BUTTON, reset_wdg)
    assert not log_wdg.GetValue()
    hist_mock.reset_mock()
