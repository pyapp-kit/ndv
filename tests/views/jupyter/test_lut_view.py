from __future__ import annotations

import cmap
from pytest import fixture

from ndv.models._lut_model import LUTModel
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

    new_clims = (4, 5)
    assert view._clims.value != new_clims
    model.clims = new_clims
    assert view._clims.value == new_clims

    new_visible = not model.visible
    model.visible = new_visible
    assert view._visible.value == new_visible

    new_cmap = cmap.Colormap("red")
    new_name = new_cmap.name.split(":")[-1]
    assert view._cmap.value != new_name
    model.cmap = new_cmap
    assert view._cmap.value == new_name

    new_autoscale = not model.autoscale
    model.autoscale = new_autoscale
    assert view._auto_clim.value == new_autoscale


def test_JupyterLutView_update_view(model: LUTModel, view: JupyterLutView) -> None:
    """Ensures the model updates when the view is changed."""

    new_clims = (5, 6)
    assert model.clims != new_clims
    view._clims.value = new_clims
    assert model.clims == new_clims

    new_visible = not model.visible
    view._visible.value = new_visible
    assert view._visible.value == new_visible

    new_cmap = view._cmap.options[1]
    assert model.cmap != new_cmap
    view._cmap.value = new_cmap
    assert model.cmap == new_cmap

    new_autoscale = not model.autoscale
    view._auto_clim.value = new_autoscale
    assert model.autoscale == new_autoscale

    # When gui clims change, autoscale should be disabled
    model.autoscale = True
    view._clims.value = (0, 1)
    assert model.autoscale is False
