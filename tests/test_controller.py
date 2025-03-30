"""Test controller without canavs or gui frontend"""

from __future__ import annotations

import os
from typing import Any, Callable, cast, no_type_check
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.controllers import ArrayViewer
from ndv.controllers._channel_controller import ChannelController
from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.models._roi_model import RectangularROIModel
from ndv.models._viewer_model import InteractionMode
from ndv.views import _app, gui_frontend
from ndv.views.bases import ArrayView, LutView
from ndv.views.bases._graphics._canvas import ArrayCanvas, HistogramCanvas
from ndv.views.bases._graphics._canvas_elements import ImageHandle

try:
    from qtpy import API_NAME
except ImportError:
    API_NAME = None

IS_WIN = os.name == "nt"
IS_PYSIDE6 = API_NAME == "PySide6"
IS_PYGFX = _app.canvas_backend(None) == "pygfx"


def _get_mock_canvas(*_: Any) -> ArrayCanvas:
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
    mock_view.hide_sliders.assert_called_once_with({2, 3}, show_remainder=True)
    # channel mode is set to default (which is currently grayscale)
    mock_view.set_channel_mode.assert_called_once_with(model.channel_mode)
    # data info is set
    mock_view.set_data_info.assert_called_once_with(wrapper.summary_info())
    model.current_index.assign({0: 1})

    # changing visible axes updates which sliders are visible
    model.visible_axes = (0, 3)
    mock_view.hide_sliders.assert_called_with({0, 3}, show_remainder=True)

    # changing the channel mode updates the sliders and updates the view combobox
    mock_view.hide_sliders.reset_mock()
    model.channel_mode = "composite"
    mock_view.set_channel_mode.assert_called_with(ChannelMode.COMPOSITE)
    mock_view.hide_sliders.assert_called_once_with(
        {0, 3, model.channel_axis}, show_remainder=True
    )
    model.channel_mode = ChannelMode.GRAYSCALE
    mock_view.hide_sliders.assert_called_with({0, 3}, show_remainder=True)

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
    ch_ctrl.lut_views[0].set_colormap.reset_mock()
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
    ctrl._add_histogram(None)
    mock_view.add_histogram.assert_called_once()
    mock_histogram = ctrl._histograms[None]
    mock_histogram.set_data.assert_called_once()

    # changing the index updates the histogram
    mock_histogram.set_data.reset_mock()
    ctrl.display_model.current_index.assign({0: 1, 1: 2, 3: 3})
    mock_histogram.set_data.assert_called_once()

    # switching to composite mode puts the histogram view in the
    # lut controller for all channels (this may change)
    ctrl.display_model.channel_mode = ChannelMode.COMPOSITE
    assert mock_histogram in ctrl._lut_controllers[None].lut_views


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


@pytest.mark.usefixtures("any_app")
def test_channel_autoscale() -> None:
    ctrl = ChannelController(key=None, lut_model=LUTModel(), views=[])

    # NB: Use a planar dataset so we can manually compute the min/max
    data = np.random.randint(0, 255, size=(10, 10), dtype="uint8")
    mi, ma = np.nanmin(data), np.nanmax(data)
    handle = MagicMock(spec=ImageHandle)
    handle.data.return_value = data
    ctrl.add_handle(handle)

    # Test some random LutController
    lut_model = ctrl.lut_model
    lut_model.clims = ClimsManual(min=1, max=2)

    # Ensure newly added lut views have the correct clims
    mock_viewer = MagicMock(LutView)
    ctrl.add_lut_view(mock_viewer)
    mock_viewer.set_clims.assert_called_once_with((1, 2))

    # Ensure autoscaling sets the clims
    mock_viewer.set_clims.reset_mock()
    lut_model.clims = ClimsMinMax()
    mock_viewer.set_clims.assert_called_once_with((mi, ma))


@pytest.mark.skipif(
    bool(IS_WIN and IS_PYSIDE6 and IS_PYGFX), reason="combo still segfaulting on CI"
)
@pytest.mark.usefixtures("any_app")
def test_array_viewer_histogram() -> None:
    """Mostly a smoke test for basic functionality of histogram backends."""

    viewer = ArrayViewer()
    viewer.show()
    viewer._add_histogram(None)
    histogram = viewer._histograms.get(None, None)
    assert histogram is not None

    # change views
    if "pygfx" not in type(histogram).__name__.lower():
        histogram.set_vertical(True)
        histogram.set_log_base(10)

    # update data
    np.random.seed(0)
    maxval = 2**16 - 1
    data = np.random.randint(0, maxval, (1000,), dtype="uint16")
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    histogram.set_data(counts, bin_edges)

    histogram.close()


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_roi_controller() -> None:
    ctrl = ArrayViewer()
    roi = RectangularROIModel()
    viewer = ctrl._viewer_model

    # Until a user interacts with ctrl.roi, there is no ROI model
    assert ctrl._roi_model is None
    ctrl.roi = roi
    assert ctrl._roi_model is not None

    # Clicking the ROI button and then clicking the canvas creates a ROI
    viewer.interaction_mode = InteractionMode.CREATE_ROI
    canvas_pos = (5, 5)
    mpe = MousePressEvent(canvas_pos[0], canvas_pos[1], MouseButton.LEFT)

    # Note - avoid diving into rendering logic here - just identify view
    with patch.object(ctrl._canvas, "elements_at", return_value=[ctrl._roi_view]):
        ctrl._canvas.on_mouse_press(mpe)
    world_pos = ctrl._canvas.canvas_to_world(canvas_pos)

    assert roi.bounding_box == (
        (world_pos[0], world_pos[1]),
        (world_pos[0] + 1, world_pos[1] + 1),
    )
    assert viewer.interaction_mode == InteractionMode.PAN_ZOOM


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_roi_interaction() -> None:
    if _app.gui_frontend() == _app.GuiFrontend.JUPYTER and IS_PYGFX:
        pytest.skip("Invalid canvas size on CI")
        return

    ctrl = ArrayViewer()
    roi = RectangularROIModel()
    ctrl.roi = roi
    roi_view = ctrl._roi_view
    assert roi_view is not None

    # FIXME: We need a large world space on the canvas, but
    # VispyArrayCanvas.set_range is not implemented yet. This workaround
    # sets the range to the extent of the data i.e. the extent of the ROI
    roi.bounding_box = ((0, 0), (500, 500))
    ctrl._canvas.set_range()
    # Note that these positions are far apart to satisfy sufficient distance
    # in world space
    canvas_roi_start = (200, 200)
    world_roi_start = tuple(ctrl._canvas.canvas_to_world(canvas_roi_start)[:2])
    canvas_new_start = (100, 100)
    world_new_start = tuple(ctrl._canvas.canvas_to_world(canvas_new_start)[:2])
    canvas_roi_end = (300, 300)
    world_roi_end = tuple(ctrl._canvas.canvas_to_world(canvas_roi_end)[:2])
    roi.bounding_box = (world_roi_start, world_roi_end)

    # Note - avoid diving into rendering logic here - just identify view
    with patch.object(ctrl._canvas, "elements_at", return_value=[ctrl._roi_view]):
        # Test moving handle
        assert not roi_view.selected()
        mpe = MousePressEvent(
            canvas_roi_start[0], canvas_roi_start[1], MouseButton.LEFT
        )
        ctrl._canvas.on_mouse_press(mpe)
        assert roi_view.selected()
        mme = MouseMoveEvent(canvas_new_start[0], canvas_new_start[1], MouseButton.LEFT)
        ctrl._canvas.on_mouse_move(mme)
        assert roi.bounding_box[0] == pytest.approx(world_new_start, 1e-6)
        assert roi.bounding_box[1] == pytest.approx(world_roi_end, 1e-6)
        mre = MouseReleaseEvent(
            canvas_new_start[0], canvas_new_start[1], MouseButton.LEFT
        )
        ctrl._canvas.on_mouse_release(mre)

        # Test translation
        roi.bounding_box = (world_roi_start, world_roi_end)
        mpe = MousePressEvent(
            (canvas_roi_start[0] + canvas_roi_end[0] / 2),
            (canvas_roi_start[1] + canvas_roi_end[1] / 2),
            MouseButton.LEFT,
        )
        ctrl._canvas.on_mouse_press(mpe)
        assert roi_view.selected()
        mme = MouseMoveEvent(
            (canvas_roi_start[0] + canvas_new_start[0] / 2),
            (canvas_roi_start[1] + canvas_new_start[1] / 2),
            MouseButton.LEFT,
        )
        ctrl._canvas.on_mouse_move(mme)
        assert roi.bounding_box[0] == pytest.approx(world_new_start, 1e-6)
        assert roi.bounding_box[1] == pytest.approx(world_roi_start, 1e-6)
        mre = MouseReleaseEvent(
            (canvas_roi_start[0] + canvas_new_start[0] / 2),
            (canvas_roi_start[1] + canvas_new_start[1] / 2),
            MouseButton.LEFT,
        )
        ctrl._canvas.on_mouse_release(mre)

    # Test cursors
    roi.bounding_box = (world_roi_start, world_roi_end)
    # Top-Left corner
    mme = MouseMoveEvent(canvas_roi_start[0], canvas_roi_start[1])
    assert roi_view.get_cursor(mme) == CursorType.FDIAG_ARROW
    # Top-Right corner
    mme = MouseMoveEvent(canvas_roi_start[0], canvas_roi_end[1])
    assert roi_view.get_cursor(mme) == CursorType.BDIAG_ARROW
    # Middle
    mme = MouseMoveEvent(
        (canvas_roi_start[0] + canvas_roi_end[0]) / 2,
        (canvas_roi_start[1] + canvas_roi_end[1]) / 2,
    )
    assert roi_view.get_cursor(mme) == CursorType.ALL_ARROW


@pytest.mark.allow_leaks
@pytest.mark.usefixtures("any_app")
def test_rgb_display_magic() -> None:
    # FIXME: Something in the QLutView is causing leaked qt widgets here.
    # Doesn't seem to be coming from the QRGBView...
    def assert_rgb_magic_works(rgb_data: np.ndarray) -> None:
        viewer = ArrayViewer(rgb_data)
        assert viewer.display_model.channel_mode == ChannelMode.RGBA
        # Note Multiple correct answers here - modulus covers both cases
        assert cast("int", viewer.display_model.channel_axis) % rgb_data.ndim == 4
        assert cast("int", viewer.display_model.visible_axes[0]) % rgb_data.ndim == 2
        assert cast("int", viewer.display_model.visible_axes[1]) % rgb_data.ndim == 3

    rgb_data = np.ones((1, 2, 3, 4, 3), dtype=np.uint8)
    assert_rgb_magic_works(rgb_data)

    rgba_data = np.ones((1, 2, 3, 4, 4), dtype=np.uint8)
    assert_rgb_magic_works(rgba_data)
