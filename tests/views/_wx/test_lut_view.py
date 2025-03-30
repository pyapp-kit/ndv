from __future__ import annotations

import cmap
import wx
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.views._wx._array_view import WxLutView


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

    auto_scale = not model.clims.is_manual
    assert view._wxwidget.auto_clim.GetValue() == auto_scale
    model.clims = ClimsManual(min=0, max=1) if auto_scale else ClimsMinMax()
    assert view._wxwidget.auto_clim.GetValue != auto_scale

    new_visible = not model.visible
    model.visible = new_visible
    assert view._wxwidget.visible.GetValue() == new_visible

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

    mi, ma = view._wxwidget.clims.GetValues()
    auto_wdg = view._wxwidget.auto_clim
    new_clims = ClimsManual(min=mi, max=ma) if auto_wdg.GetValue() else ClimsMinMax()
    view._wxwidget.auto_clim.SetValue(not new_clims.is_manual)
    processEvent(wx.EVT_TOGGLEBUTTON, auto_wdg)
    assert model.clims == new_clims

    # When gui clims change, autoscale should be disabled
    model.clims = ClimsMinMax()
    clim_wdg = view._wxwidget.clims
    clim_wdg.SetValue(0, 1)
    processEvent(wx.EVT_SLIDER, clim_wdg)
    assert model.clims == ClimsManual(min=0, max=1)  # type:ignore
