from __future__ import annotations

from typing import TYPE_CHECKING

import cmap
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.views._qt._array_view import QLutView

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(model: LUTModel, qtbot: QtBot) -> QLutView:
    view = QLutView()
    qtbot.add_widget(view.frontend_widget())
    # Set the model
    assert view.model is None
    view.model = model
    assert view.model is model
    return view


def test_QLutView_update_model(model: LUTModel, view: QLutView) -> None:
    """Ensures the view updates when the model is changed."""

    auto_scale = not model.clims.is_manual
    assert view._qwidget.auto_clim.isChecked() == auto_scale
    model.clims = ClimsManual(min=0, max=1) if auto_scale else ClimsMinMax()
    assert view._qwidget.auto_clim.isChecked() != auto_scale

    new_visible = not model.visible
    model.visible = new_visible
    assert view._qwidget.visible.isChecked() == new_visible

    new_cmap = cmap.Colormap("red")
    assert view._qwidget.cmap.currentColormap() != new_cmap
    model.cmap = new_cmap
    assert view._qwidget.cmap.currentColormap() == new_cmap


def test_QLutView_update_view(model: LUTModel, view: QLutView) -> None:
    """Ensures the model updates when the view is changed."""

    new_visible = not model.visible
    view._qwidget.visible.setChecked(new_visible)
    assert view._qwidget.visible.isChecked() == new_visible

    new_cmap = view._qwidget.cmap.itemColormap(1)
    assert model.cmap != new_cmap
    assert new_cmap is not None
    view._qwidget.cmap.setCurrentIndex(1)
    assert model.cmap == new_cmap

    mi, ma = view._qwidget.clims.value()
    new_clims = (
        ClimsManual(min=mi, max=ma)
        if view._qwidget.auto_clim.isChecked()
        else ClimsMinMax()
    )
    view._qwidget.auto_clim.setChecked(not new_clims.is_manual)
    assert model.clims == new_clims

    # When gui clims change, autoscale should be disabled
    model.clims = ClimsMinMax()
    view._qwidget.clims.setValue((0, 1))
    assert model.clims == ClimsManual(min=0, max=1)  #  type:ignore
