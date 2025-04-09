from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import cmap
from pytest import fixture
from qtpy.QtWidgets import QWidget

from ndv.models._lut_model import ClimsManual, ClimsMinMax, ClimsPercentile, LUTModel
from ndv.views._qt._array_view import QLutView
from ndv.views.bases._graphics._canvas import HistogramCanvas

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

    # Test modifying model.clims
    assert view._qwidget.auto_clim.isChecked()
    model.clims = ClimsManual(min=0, max=1)
    assert not view._qwidget.auto_clim.isChecked()
    model.clims = ClimsPercentile(min_percentile=0, max_percentile=100)
    assert view._qwidget.auto_clim.isChecked()
    model.clims = ClimsPercentile(min_percentile=1, max_percentile=99)
    assert view._qwidget.lower_tail.value() == 1
    assert view._qwidget.upper_tail.value() == 1

    # Test modifying model.visible
    assert view._qwidget.visible.isChecked()
    model.visible = False
    assert not view._qwidget.visible.isChecked()
    model.visible = True
    assert view._qwidget.visible.isChecked()

    # Test modifying model.cmap
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

    # Test toggling auto_clim
    assert model.clims == ClimsPercentile(min_percentile=0, max_percentile=100)
    view._qwidget.auto_clim.setChecked(False)
    mi, ma = view._qwidget.clims.value()
    assert model.clims == ClimsManual(min=mi, max=ma)
    view._qwidget.auto_clim.setChecked(True)
    assert model.clims == ClimsPercentile(min_percentile=0, max_percentile=100)

    # Test modifying tails changes percentiles
    view._qwidget.lower_tail.setValue(0.1)
    assert model.clims == ClimsPercentile(min_percentile=0.1, max_percentile=100)
    view._qwidget.upper_tail.setValue(0.2)
    assert model.clims == ClimsPercentile(min_percentile=0.1, max_percentile=99.8)

    # When gui clims change, autoscale should be disabled
    model.clims = ClimsMinMax()
    view._qwidget.clims.setValue((0, 1))
    assert model.clims == ClimsManual(min=0, max=1)  #  type:ignore


def test_QLutView_histogram_controls(model: LUTModel, view: QLutView) -> None:
    # Mock up a histogram
    hist_mock = MagicMock(spec=HistogramCanvas)
    hist_frontend = QWidget()
    hist_mock.frontend_widget.return_value = hist_frontend

    # Add the histogram and assert it was correctly added
    view._add_histogram(hist_mock)
    assert view.histogram is hist_mock

    # Assert histogram button toggles visibility
    # Note that the parent must be visible for child visibility to change
    old_visibility = view._qwidget.histogram_btn.isChecked()
    view._qwidget.setVisible(True)
    assert not view._qwidget.histogram_btn.isChecked()

    view._qwidget.histogram_btn.setChecked(True)
    assert view._qwidget.hist_log.isVisible()
    assert view._qwidget.hist_range.isVisible()
    assert hist_frontend.isVisible()

    view._qwidget.histogram_btn.setChecked(False)
    assert not view._qwidget.hist_log.isVisible()
    assert not view._qwidget.hist_range.isVisible()
    assert not hist_frontend.isVisible()
    view._qwidget.setVisible(old_visibility)

    # Assert toggling the log button alters the logarithmic base
    view._qwidget.hist_log.setChecked(True)
    hist_mock.set_log_base.assert_called_once_with(10)
    hist_mock.reset_mock()
    view._qwidget.hist_log.setChecked(False)
    hist_mock.set_log_base.assert_called_once_with(None)
    hist_mock.reset_mock()

    # Assert pressing the reset view button sets the histogram range
    view._qwidget.hist_range.click()
    hist_mock.set_range.assert_called_once_with()
    hist_mock.reset_mock()

    # Assert pressing the reset view button turns off log mode
    view._qwidget.hist_log.setChecked(True)
    view._qwidget.hist_range.click()
    assert not view._qwidget.hist_log.isChecked()
    hist_mock.reset_mock()
