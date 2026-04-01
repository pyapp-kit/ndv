"""Tests for shared histogram controller wiring and behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, no_type_check
from unittest.mock import MagicMock, patch

import numpy as np

from ndv.controllers import ArrayViewer
from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimsManual, ClimsMinMax
from ndv.views import _app
from ndv.views.bases import ArrayView, LUTView
from ndv.views.bases._graphics._canvas import (
    ArrayCanvas,
    HistogramCanvas,
    SharedHistogramCanvas,
)
from ndv.views.bases._graphics._canvas_elements import ImageHandle

if TYPE_CHECKING:
    from collections.abc import Callable

SHAPE = (10, 3, 10, 10)


def _make_img_handle() -> MagicMock:
    handle = MagicMock(spec=ImageHandle)
    handle.data.return_value = np.zeros((10, 10)).astype(np.uint8)
    return handle


def _get_mock_canvas(*_: Any) -> ArrayCanvas:
    mock = MagicMock(spec=ArrayCanvas)
    mock.add_image.side_effect = lambda *a, **k: _make_img_handle()
    mock.add_volume.side_effect = lambda *a, **k: _make_img_handle()
    return mock


def _get_mock_view(*_: Any) -> ArrayView:
    mock = MagicMock(spec=ArrayView)
    mock.add_lut_view.side_effect = lambda *a, **k: MagicMock(spec=LUTView)
    return mock


def _get_mock_hist_canvas() -> HistogramCanvas:
    return MagicMock(spec=HistogramCanvas)


def _get_mock_shared_hist_canvas() -> SharedHistogramCanvas:
    return MagicMock(spec=SharedHistogramCanvas)


def _patch_views(f: Callable) -> Callable:
    f = patch.object(_app, "get_array_canvas_class", lambda: _get_mock_canvas)(f)
    f = patch.object(_app, "get_array_view_class", lambda: _get_mock_view)(f)
    f = patch.object(_app, "get_histogram_canvas_class", lambda: _get_mock_hist_canvas)(f)  # fmt: skip # noqa: E501
    f = patch.object(_app, "get_shared_histogram_canvas_class", lambda: _get_mock_shared_hist_canvas)(f)  # fmt: skip # noqa: E501
    f = patch.object(_app, "filter_key_events", lambda *a, **k: lambda: None)(f)
    return f


def _make_ctrl_with_data(
    channel_mode: str = "composite",
) -> ArrayViewer:
    """Create an ArrayViewer with data loaded synchronously."""
    ctrl = ArrayViewer(channel_mode=channel_mode)
    ctrl._async = False
    ctrl.data = np.random.randint(0, 255, SHAPE, dtype=np.uint8)
    return ctrl


# ---------- Creation and connection tests ----------


@no_type_check
@_patch_views
def test_shared_histogram_creation() -> None:
    """Shared histogram is created and added to view on demand."""
    ctrl = _make_ctrl_with_data()

    assert ctrl._shared_histogram is None
    ctrl._add_shared_histogram()

    assert ctrl._shared_histogram is not None
    ctrl._view.add_shared_histogram.assert_called_once_with(ctrl._shared_histogram)


@no_type_check
@_patch_views
def test_shared_histogram_not_created_at_init() -> None:
    """use_shared_histogram=True only controls style, not visibility."""
    ctrl = ArrayViewer(viewer_options={"use_shared_histogram": True})
    ctrl._async = False
    ctrl.data = np.random.randint(0, 255, SHAPE, dtype=np.uint8)

    # Should NOT be created automatically
    assert ctrl._shared_histogram is None


@no_type_check
@_patch_views
def test_shared_histogram_idempotent() -> None:
    """Calling _add_shared_histogram twice doesn't create a second one."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    first = ctrl._shared_histogram
    ctrl._add_shared_histogram()
    assert ctrl._shared_histogram is first
    ctrl._view.add_shared_histogram.assert_called_once()


# ---------- Data flow tests ----------


@no_type_check
@_patch_views
def test_channel_data_flows_to_shared_histogram() -> None:
    """Stats updates propagate channel data to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Each channel key should appear in set_channel_data calls
    called_keys = {call[0][0] for call in mock_hist.set_channel_data.call_args_list}
    for key in ctrl._lut_controllers:
        if ctrl._lut_controllers[key].handles:
            assert key in called_keys, f"Channel {key} never got data"


@no_type_check
@_patch_views
def test_initial_state_set_on_connection() -> None:
    """Color, visibility, gamma, and name are set when channel connects."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # set_channel_color should have been called for each channel
    assert mock_hist.set_channel_color.call_count >= len(ctrl._lut_controllers)
    # set_channel_visible should have been called
    assert mock_hist.set_channel_visible.call_count >= len(ctrl._lut_controllers)
    # set_channel_gamma should have been called
    assert mock_hist.set_channel_gamma.call_count >= len(ctrl._lut_controllers)


@no_type_check
@_patch_views
def test_new_channel_connects_to_existing_shared_histogram() -> None:
    """When a new channel appears after shared histogram exists, it connects."""
    ctrl = ArrayViewer(channel_mode="composite")
    ctrl._async = False

    # Create shared histogram first (no data yet)
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram
    mock_hist.set_channel_data.reset_mock()

    # Now load data — channels get created
    ctrl.data = np.random.randint(0, 255, SHAPE, dtype=np.uint8)

    # New channels should have sent data to the shared histogram
    assert mock_hist.set_channel_data.call_count > 0


# ---------- Bidirectional sync tests ----------


@no_type_check
@_patch_views
def test_clims_from_shared_histogram_update_model() -> None:
    """Dragging clims on shared histogram updates the LUT model."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    # Pick a channel
    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    # Simulate user dragging clims on the shared histogram
    new_clims = (10.0, 200.0)
    ctrl._on_shared_histogram_clims_changed(key, new_clims)

    assert isinstance(lut_model.clims, ClimsManual)
    assert lut_model.clims.min == 10.0
    assert lut_model.clims.max == 200.0


@no_type_check
@_patch_views
def test_gamma_from_shared_histogram_updates_model() -> None:
    """Dragging gamma on shared histogram updates the LUT model."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    ctrl._on_shared_histogram_gamma_changed(key, 2.5)
    assert lut_model.gamma == 2.5


@no_type_check
@_patch_views
def test_model_clims_sync_to_shared_histogram() -> None:
    """Manual clim changes on the model propagate to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram
    mock_hist.set_channel_clims.reset_mock()

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    # Set manual clims on the model
    lut_model.clims = ClimsManual(min=50, max=150)

    # Should propagate to shared histogram
    mock_hist.set_channel_clims.assert_called_with(key, (50, 150))


@no_type_check
@_patch_views
def test_model_cmap_sync_to_shared_histogram() -> None:
    """Colormap changes on the model propagate to shared histogram."""
    import cmap

    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram
    mock_hist.set_channel_color.reset_mock()

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    lut_model.cmap = cmap.Colormap("red")
    mock_hist.set_channel_color.assert_called_once()
    call_key = mock_hist.set_channel_color.call_args[0][0]
    assert call_key == key


@no_type_check
@_patch_views
def test_model_visibility_sync_to_shared_histogram() -> None:
    """Visibility changes on the model propagate to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram
    mock_hist.set_channel_visible.reset_mock()

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    lut_model.visible = False
    mock_hist.set_channel_visible.assert_called_with(key, False)

    lut_model.visible = True
    mock_hist.set_channel_visible.assert_called_with(key, True)


# ---------- Autoscale sync tests ----------


@no_type_check
@_patch_views
def test_autoscale_syncs_clims_to_shared_histogram() -> None:
    """When autoscale recomputes clims, shared histogram clim lines update."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    # First set manual clims
    lut_model.clims = ClimsManual(min=10, max=200)
    mock_hist.set_channel_clims.reset_mock()

    # Switch back to autoscale
    lut_model.clims = ClimsMinMax()

    # Should have synced resolved clims
    assert mock_hist.set_channel_clims.call_count >= 1


@no_type_check
@_patch_views
def test_stats_update_syncs_clims() -> None:
    """When new data arrives, resolved clims from stats reach the histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram
    mock_hist.set_channel_clims.reset_mock()

    # Trigger a data update by changing index
    ctrl.display_model.current_index.assign({0: 1})

    # Should have called set_channel_clims with resolved values
    assert mock_hist.set_channel_clims.call_count >= 1


# ---------- Channel mode visibility tests ----------


@no_type_check
@_patch_views
def test_grayscale_mode_shows_only_default_channel() -> None:
    """In grayscale mode, only key=None channel is visible on shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Switch to grayscale
    ctrl.display_model.channel_mode = ChannelMode.GRAYSCALE
    mock_hist.set_channel_visible.reset_mock()

    # Trigger visibility update again to capture calls
    ctrl._update_lut_visibility(ChannelMode.GRAYSCALE)

    # key=None should be visible, numbered keys should be hidden
    calls = {
        args[0][0]: args[0][1] for args in mock_hist.set_channel_visible.call_args_list
    }
    assert calls.get(None) is True
    for key in ctrl._lut_controllers:
        if key is not None:
            assert calls.get(key) is False


@no_type_check
@_patch_views
def test_composite_mode_shows_numbered_channels() -> None:
    """In composite mode, numbered channels are visible."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Ensure composite mode
    ctrl.display_model.channel_mode = ChannelMode.COMPOSITE
    mock_hist.set_channel_visible.reset_mock()

    ctrl._update_lut_visibility(ChannelMode.COMPOSITE)

    calls = {
        args[0][0]: args[0][1] for args in mock_hist.set_channel_visible.call_args_list
    }
    for key in ctrl._lut_controllers:
        if key is not None and key != "RGB":
            assert calls.get(key) is True


@no_type_check
@_patch_views
def test_opening_histogram_in_grayscale_respects_mode() -> None:
    """If histogram is opened while in grayscale, only default channel shows."""
    ctrl = _make_ctrl_with_data()

    # Switch to grayscale before opening histogram
    ctrl.display_model.channel_mode = ChannelMode.GRAYSCALE

    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Check the last set_channel_visible call for each key
    visible_calls: dict[Any, bool] = {}
    for call in mock_hist.set_channel_visible.call_args_list:
        visible_calls[call[0][0]] = call[0][1]

    assert visible_calls.get(None) is True
    for key in ctrl._lut_controllers:
        if key is not None:
            assert visible_calls.get(key) is False


# ---------- Clim bounds tests ----------


@no_type_check
@_patch_views
def test_clim_bounds_propagate_to_shared_histogram() -> None:
    """clim_bounds from LUT model are forwarded to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # For uint8 data, clim_bounds should be (0, 255)
    mock_hist.set_clim_bounds.assert_called()


# ---------- Highlight tests ----------


@no_type_check
@_patch_views
def test_highlight_forwarded_to_shared_histogram() -> None:
    """Mouse hover values are forwarded to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    ctrl._highlight_values({None: 42.0}, (5, 5))
    mock_hist.highlight.assert_called_once_with({None: 42.0})


@no_type_check
@_patch_views
def test_highlight_clears_on_empty_values() -> None:
    """Empty channel values forward None to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    ctrl._highlight_values({}, (5, 5))
    mock_hist.highlight.assert_called_with({})
