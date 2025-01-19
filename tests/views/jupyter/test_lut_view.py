from __future__ import annotations

import cmap
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.views._jupyter._array_view import JupyterLutView


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(model: LUTModel) -> JupyterLutView:
    view = JupyterLutView()
    # Set the model
    assert view.model is None
    view.model = model
    assert view.model is model
    return view


def test_JupyterLutView_update_model(model: LUTModel, view: JupyterLutView) -> None:
    """Ensures the view updates when the model is changed."""

    auto_scale = not model.clims.is_manual
    assert view._auto_clim.value == auto_scale
    model.clims = ClimsManual(min=0, max=1) if auto_scale else ClimsMinMax()
    assert view._auto_clim.value != auto_scale

    new_visible = not model.visible
    model.visible = new_visible
    assert view._visible.value == new_visible

    new_cmap = cmap.Colormap("red")
    new_name = new_cmap.name.split(":")[-1]
    assert view._cmap.value != new_name
    model.cmap = new_cmap
    assert view._cmap.value == new_name


def test_JupyterLutView_update_view(model: LUTModel, view: JupyterLutView) -> None:
    """Ensures the model updates when the view is changed."""

    new_visible = not model.visible
    view._visible.value = new_visible
    assert view._visible.value == new_visible

    new_cmap = view._cmap.options[1]
    assert model.cmap != new_cmap
    view._cmap.value = new_cmap
    assert model.cmap == new_cmap

    mi, ma = view._clims.value
    new_clims = ClimsManual(min=mi, max=ma) if view._auto_clim.value else ClimsMinMax()
    view._auto_clim.value = not new_clims.is_manual
    assert model.clims == new_clims

    # When gui clims change, autoscale should be disabled
    model.clims = ClimsMinMax()
    view._clims.value = (0, 1)
    assert model.clims == ClimsManual(min=0, max=1)  # type:ignore
