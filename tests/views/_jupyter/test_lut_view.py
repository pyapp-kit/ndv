from __future__ import annotations

from unittest.mock import MagicMock

import cmap
from jupyter_rfb.widget import RemoteFrameBuffer
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.views._jupyter._array_view import JupyterLutView
from ndv.views.bases._graphics._canvas import HistogramCanvas


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


def test_JupyterLutView_histogram_controls(view: JupyterLutView) -> None:
    # Mock up a histogram
    hist_mock = MagicMock(spec=HistogramCanvas)
    hist_frontend = RemoteFrameBuffer()
    hist_mock.frontend_widget.return_value = hist_frontend

    # Add the histogram and assert it was correctly added
    view.add_histogram(hist_mock)
    assert view._histogram is hist_mock

    # Assert histogram button toggles visibility
    view._histogram_btn.value = True
    assert view._histogram_container.layout.display == "flex"
    view._histogram_btn.value = False
    assert view._histogram_container.layout.display == "none"

    # Assert toggling the log button alters the logarithmic base
    view._log.value = True
    hist_mock.set_log_base.assert_called_once_with(10)
    hist_mock.reset_mock()

    view._log.value = False
    hist_mock.set_log_base.assert_called_once_with(None)
    hist_mock.reset_mock()

    # Assert pressing the reset view button sets the histogram range
    view._reset_histogram.click()
    hist_mock.set_range.assert_called_once_with()
    hist_mock.reset_mock()

    # Assert pressing the reset view button turns off log mode
    view._log.value = True
    view._reset_histogram.click()
    assert not view._log.value
    hist_mock.reset_mock()
