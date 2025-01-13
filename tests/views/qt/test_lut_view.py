from __future__ import annotations

import cmap
from pytest import fixture

from ndv.models._lut_model import LUTModel
from ndv.views._app import QtProvider
from ndv.views._qt._array_view import QLutView


@fixture(autouse=True)
def init_provider() -> None:
    provider = QtProvider()
    provider.create_app()


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(model: LUTModel) -> QLutView:
    view = QLutView()
    # Set the model
    assert view.model is None
    view.model = model
    assert view.model is model
    return view


def test_QLutView_update_model(model: LUTModel, view: QLutView) -> None:
    """Ensures the view updates when the model is changed."""

    new_clims = (4, 5)
    assert view._qwidget.clims.value() != new_clims
    model.clims = new_clims
    assert view._qwidget.clims.value() == new_clims

    new_visible = not model.visible
    model.visible = new_visible
    assert view._qwidget.visible.isChecked() == new_visible

    new_cmap = cmap.Colormap("red")
    assert view._qwidget.cmap.currentColormap() != new_cmap
    model.cmap = new_cmap
    assert view._qwidget.cmap.currentColormap() == new_cmap

    new_autoscale = not model.autoscale
    model.autoscale = new_autoscale
    assert view._qwidget.auto_clim.isChecked() == new_autoscale


def test_QLutView_update_view(model: LUTModel, view: QLutView) -> None:
    """Ensures the model updates when the view is changed."""

    new_clims = (5, 6)
    assert model.clims != new_clims
    view._qwidget.clims.setValue(new_clims)
    assert model.clims == new_clims

    new_visible = not model.visible
    view._qwidget.visible.setChecked(new_visible)
    assert view._qwidget.visible.isChecked() == new_visible

    new_cmap = view._qwidget.cmap.itemColormap(1)
    assert model.cmap != new_cmap
    assert new_cmap is not None
    view._qwidget.cmap.setCurrentIndex(1)
    assert model.cmap == new_cmap

    new_autoscale = not model.autoscale
    view._qwidget.auto_clim.setChecked(new_autoscale)
    assert model.autoscale == new_autoscale

    # When gui clims change, autoscale should be disabled
    model.autoscale = True
    view._qwidget.clims.setValue((0, 1))
    assert model.autoscale is False
