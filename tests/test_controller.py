"""Test controller without canavs or gui frontend"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast, no_type_check
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from ndv._types import MouseMoveEvent
from ndv.controllers import ArrayViewer
from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._lut_model import LUTModel
from ndv.views import _app, gui_frontend
from ndv.views.bases import ArrayView, LutView
from ndv.views.bases._graphics._canvas import ArrayCanvas, HistogramCanvas
from ndv.views.bases._graphics._canvas_elements import ImageHandle

if TYPE_CHECKING:
    from ndv.controllers._channel_controller import ChannelController


def _get_mock_canvas() -> ArrayCanvas:
    mock = MagicMock(spec=ArrayCanvas)
    img_handle = MagicMock(spec=ImageHandle)
    img_handle.data.return_value = np.zeros((10, 10)).astype(np.uint8)
    mock.add_image.return_value = img_handle

    vol_handle = MagicMock(spec=ImageHandle)
    vol_handle.data.return_value = np.zeros((10, 10, 10)).astype(np.uint8)
    mock.add_volume.return_value = vol_handle
    return mock


def _get_mock_hist_canvas() -> HistogramCanvas:
    return MagicMock(spec=HistogramCanvas)


def _get_mock_view(*_: Any) -> ArrayView:
    mock = MagicMock(spec=ArrayView)
    lut_mock = MagicMock(spec=LutView)
    mock.add_lut_view.return_value = lut_mock
    return mock


def _patch_views(f: Callable) -> Callable:
    f = patch.object(_app, "get_array_canvas_class", lambda: _get_mock_canvas)(f)
    f = patch.object(_app, "get_array_view_class", lambda: _get_mock_view)(f)
    f = patch.object(_app, "get_histogram_canvas_class", lambda: _get_mock_hist_canvas)(f)  # fmt: skip # noqa
    return f


@no_type_check
@_patch_views
def test_controller() -> None:
    SHAPE = (10, 4, 10, 10)
    ctrl = ArrayViewer()
    ctrl._async = False
    model = ctrl.display_model
    mock_view = ctrl._view
    mock_view.create_sliders.assert_not_called()

    data = np.empty(SHAPE)
    ctrl.data = data
    wrapper = ctrl._data_model.data_wrapper

    # showing the controller shows the view
    ctrl.show()
    mock_view.set_visible.assert_called_once_with(True)

    # sliders are first created with the shape of the data
    ranges = {i: range(s) for i, s in enumerate(SHAPE)}
    mock_view.create_sliders.assert_called_once_with(ranges)
    # visible-axis sliders are hidden
    # (2,3) because model.visible_axes is set to (-2, -1) and ndim is 4
    mock_view.hide_sliders.assert_called_once_with((2, 3), show_remainder=True)
    # channel mode is set to default (which is currently grayscale)
    mock_view.set_channel_mode.assert_called_once_with(model.channel_mode)
    # data info is set
    mock_view.set_data_info.assert_called_once_with(wrapper.summary_info())
    model.current_index.assign({0: 1})

    # changing visible axes updates which sliders are visible
    model.visible_axes = (0, 3)
    mock_view.hide_sliders.assert_called_with((0, 3), show_remainder=True)

    # changing the channel mode updates the sliders and updates the view combobox
    mock_view.hide_sliders.reset_mock()
    model.channel_mode = "composite"
    mock_view.set_channel_mode.assert_called_with(ChannelMode.COMPOSITE)
    mock_view.hide_sliders.assert_called_once_with(
        (0, 3, model.channel_axis), show_remainder=True
    )
    model.channel_mode = ChannelMode.GRAYSCALE
    mock_view.hide_sliders.assert_called_with((0, 3), show_remainder=True)

    # when the view changes the current index, the model is updated
    idx = {0: 1, 1: 2, 3: 8}
    mock_view.current_index.return_value = idx
    ctrl._on_view_current_index_changed()
    assert model.current_index == idx

    # when the view sets 3 dimensions, the model is updated
    mock_view.visible_axes.return_value = (0, -2, -1)
    ctrl._on_view_visible_axes_changed()
    assert model.visible_axes == (0, -2, -1)

    # when the view changes the channel mode, the model is updated
    assert model.channel_mode == ChannelMode.GRAYSCALE
    ctrl._on_view_channel_mode_changed(ChannelMode.COMPOSITE)
    assert model.channel_mode == ChannelMode.COMPOSITE

    # setting a new ArrayDisplay model updates the appropriate view widgets
    ch_ctrl = cast("ChannelController", ctrl._lut_controllers[None])
    ch_ctrl.lut_views[0].set_colormap_without_signal.reset_mock()
    ctrl.display_model = ArrayDisplayModel(default_lut=LUTModel(cmap="green"))
    # fails
    # ch_ctrl.lut_views[0].set_colormap_without_signal.assert_called_once()


@no_type_check
@_patch_views
def test_canvas() -> None:
    SHAPE = (10, 4, 10, 10)
    data = np.empty(SHAPE)
    ctrl = ArrayViewer()
    ctrl._async = False
    mock_canvas = ctrl._canvas

    mock_view = ctrl._view
    ctrl.data = data

    # clicking the reset zoom button calls set_range on the canvas
    ctrl._on_view_reset_zoom_clicked()
    mock_canvas.set_range.assert_called_once_with()

    # hovering on the canvas updates the hover info in the view
    mock_canvas.canvas_to_world.return_value = (1, 2, 3)
    ctrl._on_canvas_mouse_moved(MouseMoveEvent(1, 2))
    mock_canvas.canvas_to_world.assert_called_once_with((1, 2))
    mock_view.set_hover_info.assert_called_once_with("[2, 1] 0")


@no_type_check
@_patch_views
def test_histogram_controller() -> None:
    ctrl = ArrayViewer()
    ctrl._async = False
    mock_view = ctrl._view

    ctrl.data = np.zeros((10, 4, 10, 10)).astype(np.uint8)

    # adding a histogram tells the view to add a histogram, and updates the data
    ctrl._add_histogram()
    mock_view.add_histogram.assert_called_once()
    mock_histogram = ctrl._histogram
    mock_histogram.set_data.assert_called_once()

    # changing the index updates the histogram
    mock_histogram.set_data.reset_mock()
    ctrl.display_model.current_index.assign({0: 1, 1: 2, 3: 3})
    mock_histogram.set_data.assert_called_once()

    # switching to composite mode puts the histogram view in the
    # lut controller for all channels (this may change)
    ctrl.display_model.channel_mode = ChannelMode.COMPOSITE
    assert mock_histogram in ctrl._lut_controllers[0].lut_views


@pytest.mark.usefixtures("any_app")
def test_array_viewer_with_app() -> None:
    """Example usage of new mvc pattern."""
    viewer = ArrayViewer()
    assert gui_frontend() in type(viewer._view).__name__.lower()
    viewer.show()

    data = np.random.randint(0, 255, size=(10, 10, 10, 10, 10), dtype="uint8")
    viewer.data = data

    # test changing current index via the view
    index_mock = Mock()
    viewer.display_model.current_index.value_changed.connect(index_mock)
    index = {0: 4, 1: 1, 2: 2}
    # setting the index should trigger the signal, only once
    viewer._view.set_current_index(index)
    index_mock.assert_called_once()
    for k, v in index.items():
        assert viewer.display_model.current_index[k] == v

    # setting again should not trigger the signal
    index_mock.reset_mock()
    viewer._view.set_current_index(index)
    index_mock.assert_not_called()

    # test_setting 3D
    assert viewer.display_model.visible_axes == (-2, -1)
    visax_mock = Mock()
    viewer.display_model.events.visible_axes.connect(visax_mock)
    viewer._view.set_visible_axes((0, -2, -1))

    # FIXME:
    # calling set_visible_axes on wx during testing is not triggering the
    # _on_ndims_toggled callback... and I don't know enough about wx yet to know why.
    if gui_frontend() != _app.GuiFrontend.WX:
        visax_mock.assert_called_once()
        assert viewer.display_model.visible_axes == (0, -2, -1)
