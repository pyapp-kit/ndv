from __future__ import annotations

from typing import TYPE_CHECKING

import cmap
import wx
from pytest import fixture

from ndv.models._lut_model import LUTModel
from ndv.views._app import WxProvider
from ndv.views._wx._array_view import WxLutView

if TYPE_CHECKING:
    from collections.abc import Generator


@fixture(autouse=True)
def app() -> Generator[None, None, None]:
    # Create wx app
    provider = WxProvider()
    provider.create_app()
    # NB: Keep app alive during test
    yield
    return


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(model: LUTModel) -> WxLutView:
    frame = wx.Frame(None)
    view = WxLutView(frame)
    assert view.model is None
    view.model = model
    assert view.model is model
    return view


def test_WxLutView_update_model(model: LUTModel, view: WxLutView) -> None:
    """Ensures the view updates when the model is changed."""

    new_clims = (4, 5)
    assert view._wxwidget.clims.GetValues() != new_clims
    model.clims = new_clims
    assert view._wxwidget.clims.GetValues() == new_clims

    new_visible = not model.visible
    model.visible = new_visible
    assert view._wxwidget.visible.GetValue() == new_visible

    new_cmap = cmap.Colormap("red")
    assert view._wxwidget.cmap.GetValue() != new_cmap
    model.cmap = new_cmap
    assert view._wxwidget.cmap.GetValue() == new_cmap

    new_autoscale = not model.autoscale
    model.autoscale = new_autoscale
    assert view._wxwidget.auto_clim.GetValue() == new_autoscale


def test_WxLutView_update_view(model: LUTModel, view: WxLutView) -> None:
    """Ensures the model updates when the view is changed."""

    new_clims = (5, 6)
    assert model.clims != new_clims
    view._wxwidget.clims.SetValue(*new_clims)
    ev = wx.PyCommandEvent(wx.EVT_SLIDER.typeId, view._wxwidget.clims.GetId())
    wx.PostEvent(view._wxwidget.clims.GetEventHandler(), ev)
    wx.Yield()
    assert model.clims == new_clims

    new_visible = not model.visible
    view._wxwidget.visible.SetValue(new_visible)
    ev = wx.PyCommandEvent(wx.EVT_CHECKBOX.typeId, view._wxwidget.visible.GetId())
    wx.PostEvent(view._wxwidget.visible.GetEventHandler(), ev)
    wx.Yield()
    assert model.visible == new_visible

    new_cmap = cmap.Colormap("red")
    assert model.cmap != new_cmap
    view._wxwidget.cmap.SetValue(new_cmap.name)
    ev = wx.PyCommandEvent(wx.EVT_COMBOBOX.typeId, view._wxwidget.cmap.GetId())
    wx.PostEvent(view._wxwidget.cmap.GetEventHandler(), ev)
    wx.Yield()
    assert model.cmap == new_cmap

    new_autoscale = not model.autoscale
    view._wxwidget.auto_clim.SetValue(new_autoscale)
    ev = wx.PyCommandEvent(wx.EVT_TOGGLEBUTTON.typeId, view._wxwidget.auto_clim.GetId())
    wx.PostEvent(view._wxwidget.auto_clim.GetEventHandler(), ev)
    wx.Yield()
    assert model.autoscale == new_autoscale

    # When gui clims change, autoscale should be disabled
    model.autoscale = True
    view._wxwidget.clims.SetValue(0, 1)
    ev = wx.PyCommandEvent(wx.EVT_SLIDER.typeId, view._wxwidget.clims.GetId())
    wx.PostEvent(view._wxwidget.clims.GetEventHandler(), ev)
    wx.Yield()
    assert model.autoscale is False
