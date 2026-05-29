"""Tests for shared histogram controller wiring and behavior.

TODO: Move this to tests/views
"""

from __future__ import annotations

from typing import Any, no_type_check
from unittest.mock import patch

import numpy as np
import pytest
from pytest import fixture
from scenex.app.events import (
    MouseButton,
    MouseDoublePressEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)

from ndv.controllers import ArrayViewer
from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimsManual, ClimsMinMax
from ndv.views._shared_histogram import SharedHistogram

SHAPE = (10, 3, 10, 10)


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
@pytest.mark.usefixtures("any_app")
def test_shared_histogram_creation() -> None:
    """Shared histogram is created and added to view on demand."""
    ctrl = _make_ctrl_with_data()

    assert ctrl._shared_histogram is None
    with patch.object(ctrl._view, "add_shared_histogram") as mock_add_hist:
        ctrl._add_shared_histogram()

    assert ctrl._shared_histogram is not None
    mock_add_hist.assert_called_once_with(ctrl._shared_histogram.widget())


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_shared_histogram_not_created_at_init() -> None:
    """use_shared_histogram=True only controls style, not visibility."""
    ctrl = ArrayViewer(viewer_options={"use_shared_histogram": True})
    ctrl._async = False
    ctrl.data = np.random.randint(0, 255, SHAPE, dtype=np.uint8)

    # Should NOT be created automatically
    assert ctrl._shared_histogram is None


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_shared_histogram_idempotent() -> None:
    """Calling _add_shared_histogram twice doesn't create a second one."""
    ctrl = _make_ctrl_with_data()
    with patch.object(ctrl._view, "add_shared_histogram") as mock_add_hist:
        ctrl._add_shared_histogram()
        first = ctrl._shared_histogram
        ctrl._add_shared_histogram()
    assert ctrl._shared_histogram is first
    mock_add_hist.assert_called_once()


# ---------- Data flow tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_channel_data_flows_to_shared_histogram() -> None:
    """Stats updates propagate channel data to shared histogram."""
    with patch(
        "ndv.controllers._array_viewer.SharedHistogram.set_channel_data"
    ) as mock_set_channel_data:
        ctrl = _make_ctrl_with_data()
        ctrl._add_shared_histogram()

    # Each channel key should appear in set_channel_data calls
    called_keys = {call[0][0] for call in mock_set_channel_data.call_args_list}
    for key in ctrl._lut_controllers:
        if ctrl._lut_controllers[key].handles:
            assert key in called_keys, f"Channel {key} never got data"


@no_type_check
@patch("ndv.controllers._array_viewer.SharedHistogram.set_channel_gamma")
@patch("ndv.controllers._array_viewer.SharedHistogram.set_channel_visible")
@patch("ndv.controllers._array_viewer.SharedHistogram.set_channel_color")
@pytest.mark.usefixtures("any_app")
def test_initial_state_set_on_connection(mock_color, mock_visible, mock_gamma) -> None:
    """Color, visibility, gamma, and name are set when channel connects."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    # set_channel_color should have been called for each channel
    assert mock_color.call_count >= len(ctrl._lut_controllers)
    # set_channel_visible should have been called
    assert mock_visible.call_count >= len(ctrl._lut_controllers)
    # set_channel_gamma should have been called
    assert mock_gamma.call_count >= len(ctrl._lut_controllers)


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_new_channel_connects_to_existing_shared_histogram() -> None:
    """When a new channel appears after shared histogram exists, it connects."""
    ctrl = ArrayViewer(channel_mode="composite")
    ctrl._async = False

    # Create shared histogram first (no data yet)
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Now load data — channels get created
    with patch.object(mock_hist, "set_channel_data") as mock_set_channel_data:
        ctrl.data = np.random.randint(0, 255, SHAPE, dtype=np.uint8)

    # New channels should have sent data to the shared histogram
    assert mock_set_channel_data.call_count == len(ctrl._lut_controllers)


# ---------- Bidirectional sync tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
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
@pytest.mark.usefixtures("any_app")
def test_gamma_from_shared_histogram_updates_model() -> None:
    """Dragging gamma on shared histogram updates the LUT model."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    ctrl._on_shared_histogram_gamma_changed(key, 2.5)
    assert lut_model.gamma == 2.5


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_model_clims_sync_to_shared_histogram() -> None:
    """Manual clim changes on the model propagate to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    # Set manual clims on the model
    with patch.object(mock_hist, "set_channel_clims") as mock_set_channel_clims:
        lut_model.clims = ClimsManual(min=50, max=150)

    # Should propagate to shared histogram
    mock_set_channel_clims.assert_called_once_with(key, (50, 150))


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_model_cmap_sync_to_shared_histogram() -> None:
    """Colormap changes on the model propagate to shared histogram."""
    import cmap

    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    with patch.object(mock_hist, "set_channel_color") as mock_set_channel_color:
        lut_model.cmap = cmap.Colormap("red")
    mock_set_channel_color.assert_called_once()
    call_key = mock_set_channel_color.call_args[0][0]
    assert call_key == key


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_model_visibility_sync_to_shared_histogram() -> None:
    """Visibility changes on the model propagate to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    with patch.object(mock_hist, "set_channel_visible") as mock_set_channel_visible:
        lut_model.visible = False
    mock_set_channel_visible.assert_called_once_with(key, False)

    with patch.object(mock_hist, "set_channel_visible") as mock_set_channel_visible:
        lut_model.visible = True
    mock_set_channel_visible.assert_called_once_with(key, True)


# ---------- Autoscale sync tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_autoscale_syncs_clims_to_shared_histogram() -> None:
    """When autoscale recomputes clims, shared histogram clim lines update."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    key = next(iter(ctrl._lut_controllers))
    lut_model = ctrl._lut_controllers[key].lut_model

    # First set manual clims
    lut_model.clims = ClimsManual(min=10, max=200)

    # Switch back to autoscale
    with patch.object(mock_hist, "set_channel_clims") as mock_set_channel_clims:
        lut_model.clims = ClimsMinMax()
    mock_set_channel_clims.assert_called_once()


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_stats_update_syncs_clims() -> None:
    """When new data arrives, resolved clims from stats reach the histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Trigger a data update by changing index
    with patch.object(mock_hist, "set_channel_clims") as mock_set_channel_clims:
        ctrl.display_model.current_index.assign({0: 1})

    # Should have called set_channel_clims with resolved values
    assert mock_set_channel_clims.call_count == SHAPE[1]


# ---------- Channel mode visibility tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_grayscale_mode_shows_only_default_channel() -> None:
    """In grayscale mode, only key=None channel is visible on shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Switch to grayscale
    ctrl.display_model.channel_mode = ChannelMode.GRAYSCALE

    # Trigger visibility update again to capture calls
    with patch.object(mock_hist, "set_channel_visible") as mock_set_channel_visible:
        ctrl._update_lut_visibility(ChannelMode.GRAYSCALE)

    # key=None should be visible, numbered keys should be hidden
    calls = {args[0][0]: args[0][1] for args in mock_set_channel_visible.call_args_list}
    assert calls.get(None) is True
    for key in ctrl._lut_controllers:
        if key is not None:
            assert calls.get(key) is False


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_composite_mode_shows_numbered_channels() -> None:
    """In composite mode, numbered channels are visible."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()
    mock_hist = ctrl._shared_histogram

    # Ensure composite mode
    ctrl.display_model.channel_mode = ChannelMode.COMPOSITE

    with patch.object(mock_hist, "set_channel_visible") as mock_set_channel_visible:
        ctrl._update_lut_visibility(ChannelMode.COMPOSITE)

    calls = {args[0][0]: args[0][1] for args in mock_set_channel_visible.call_args_list}
    for key in ctrl._lut_controllers:
        if key is not None and key != "RGB":
            assert calls.get(key) is True


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_opening_histogram_in_grayscale_respects_mode() -> None:
    """If histogram is opened while in grayscale, only default channel shows."""
    ctrl = _make_ctrl_with_data()

    # Switch to grayscale before opening histogram
    ctrl.display_model.channel_mode = ChannelMode.GRAYSCALE

    with patch(
        "ndv.controllers._array_viewer.SharedHistogram.set_channel_visible"
    ) as mock_set_channel_visible:
        ctrl._add_shared_histogram()

    # Check the last set_channel_visible call for each key
    visible_calls: dict[Any, bool] = {}
    for call in mock_set_channel_visible.call_args_list:
        visible_calls[call[0][0]] = call[0][1]

    assert visible_calls.get(None) is True
    for key in ctrl._lut_controllers:
        if key is not None:
            assert visible_calls.get(key) is False


# ---------- Clim bounds tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_clim_bounds_propagate_to_shared_histogram() -> None:
    """clim_bounds from LUT model are forwarded to shared histogram."""
    ctrl = _make_ctrl_with_data()
    with patch(
        "ndv.controllers._array_viewer.SharedHistogram.set_clim_bounds"
    ) as mock_set_clim_bounds:
        ctrl._add_shared_histogram()

    # For uint8 data, clim_bounds should be (0, 255)
    mock_set_clim_bounds.assert_called()


# ---------- Highlight tests ----------


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_highlight_forwarded_to_shared_histogram() -> None:
    """Mouse hover values are forwarded to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    with patch.object(ctrl._shared_histogram, "highlight") as mock_highlight:
        ctrl._highlight_values({None: 42.0}, (5, 5))
    mock_highlight.assert_called_once_with({None: 42.0})


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_highlight_clears_on_empty_values() -> None:
    """Empty channel values forward None to shared histogram."""
    ctrl = _make_ctrl_with_data()
    ctrl._add_shared_histogram()

    with patch.object(ctrl._shared_histogram, "highlight") as mock_highlight:
        ctrl._highlight_values({}, (5, 5))
    mock_highlight.assert_called_with({})


# ---------- Visual behavior tests ----------


@fixture
def hist() -> SharedHistogram:
    canvas = SharedHistogram()
    canvas.set_range(x=(0, 100), y=(0, 1))
    return canvas


@pytest.mark.usefixtures("any_app")
def test_channel_creation(hist: SharedHistogram) -> None:
    """Channels are created lazily on first data/color set."""
    assert len(hist._channels) == 0

    counts = np.array([1, 2, 3, 2, 1])
    edges = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    hist.set_channel_data(0, counts, edges)
    assert 0 in hist._channels

    hist.set_channel_data(1, counts, edges)
    assert 1 in hist._channels
    assert len(hist._channels) == 2


@pytest.mark.usefixtures("any_app")
def test_channel_removal(hist: SharedHistogram) -> None:
    """Removed channels are cleaned up."""
    counts = np.array([1, 2, 3])
    edges = np.array([0, 33, 66, 100], dtype=float)
    hist.set_channel_data(0, counts, edges)
    hist.set_channel_data(1, counts, edges)

    hist.remove_channel(0)
    assert 0 not in hist._channels
    assert 1 in hist._channels


@pytest.mark.usefixtures("any_app")
def test_channel_visibility(hist: SharedHistogram) -> None:
    """Channel visibility controls all visual elements."""
    counts = np.array([1, 2, 3])
    edges = np.array([0, 33, 66, 100], dtype=float)
    hist.set_channel_data(0, counts, edges)
    hist.set_channel_clims(0, (10, 90))

    ch = hist._channels[0]
    assert ch.visible is True
    assert ch.area_mesh.visible is True
    assert ch.outline.visible is True

    hist.set_channel_visible(0, False)
    assert ch.visible is False
    assert ch.area_mesh.visible is False
    assert ch.outline.visible is False
    assert ch.left_clim.visible is False
    assert ch.right_clim.visible is False
    assert ch.gamma_line.visible is False
    assert ch.gamma_handle.visible is False

    hist.set_channel_visible(0, True)
    assert ch.visible is True
    assert ch.area_mesh.visible is True


@pytest.mark.usefixtures("any_app")
def test_none_key_channel() -> None:
    """key=None (grayscale default channel) works correctly."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(None, counts, edges)

    assert None in hist._channels
    ch = hist._channels[None]
    assert ch.visible is True

    hist.set_channel_clims(None, (20, 80))
    assert ch.clims == (20, 80)

    hist.set_channel_visible(None, False)
    assert ch.visible is False


@pytest.mark.usefixtures("any_app")
def test_clim_drag_emits_signal() -> None:
    """Dragging a clim handle emits climsChanged with correct key."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(0, 100))
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (0, 1, 0, 1))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    left_clim_pos = (x_start + (width / 10), y_start + height - 1)
    hist.canvas.handle(MousePressEvent(pos=left_clim_pos, buttons=MouseButton.LEFT))
    new_left_clim_pos = (x_start + (width * 3 / 10), y_start + height - 1)
    hist.canvas.handle(MouseMoveEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT)
    )

    assert len(received) >= 1
    assert received[-1][0] == 0
    assert received[-1][1][0] >= 25


@pytest.mark.usefixtures("any_app")
def test_none_key_clim_drag() -> None:
    """Clim dragging works for key=None (grayscale channel)."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(None, counts, edges)
    hist.set_range(x=(0, 100))
    hist.set_channel_clims(None, (10, 90))
    hist.set_channel_color(None, (0.5, 0.5, 0.5, 1))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    right_clim_pos = (x_start + (width * 9 / 10), y_start + height - 1)
    hist.canvas.handle(MousePressEvent(pos=right_clim_pos, buttons=MouseButton.LEFT))
    new_right_clim_pos = (x_start + (width * 7 / 10), y_start + height - 1)
    hist.canvas.handle(MouseMoveEvent(pos=new_right_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_right_clim_pos, buttons=MouseButton.LEFT)
    )

    assert len(received) >= 1
    assert received[-1][0] is None


@pytest.mark.usefixtures("any_app")
def test_gamma_double_click_resets() -> None:
    """Double-clicking gamma handle emits gammaChanged with 1.0."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(0, 100))
    hist.set_channel_clims(0, (0, 100))
    hist.set_channel_gamma(0, 2.0)
    hist.set_channel_color(0, (0, 1, 0, 1))

    received: list = []
    hist.gammaChanged.connect(lambda key, gamma: received.append((key, gamma)))

    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    gamma_height = 2 ** (-2.0)
    gamma_pos = (x_start + (width / 2), y_start + height * (1 - gamma_height))
    hist.canvas.handle(MouseDoublePressEvent(pos=gamma_pos, buttons=MouseButton.LEFT))

    assert len(received) == 1
    assert received[0] == (0, 1.0)


@pytest.mark.usefixtures("any_app")
def test_clim_bounds_constrain_drag() -> None:
    """Clim drag respects clim_bounds."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(-50, 150))
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (1, 0, 0, 1))
    hist.set_clim_bounds((0, 255))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    left_clim_pos = (x_start + (6 * width / 20), y_start + height - 1)
    hist.canvas.handle(MousePressEvent(pos=left_clim_pos, buttons=MouseButton.LEFT))
    new_left_clim_pos = (x_start + (2 * width / 20), y_start + height - 1)
    hist.canvas.handle(MouseMoveEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT)
    )

    if received:
        assert received[-1][1][0] >= 0
    hist.climsChanged.disconnect()


@pytest.mark.usefixtures("any_app")
def test_log_scale() -> None:
    """Log scale can be toggled without errors."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)

    hist.set_log_base(10)
    assert hist._log_base == 10

    hist.set_log_base(None)
    assert hist._log_base is None


@pytest.mark.usefixtures("any_app")
def test_highlight() -> None:
    """Highlight line shows and hides correctly."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    for channel in hist._channels.values():
        assert not channel.highlight.visible

    hist.highlight({"ch0": 50})
    for key, channel in hist._channels.items():
        assert channel.highlight.visible == (key == "ch0")

    hist.highlight({})
    for channel in hist._channels.values():
        assert not channel.highlight.visible


@pytest.mark.usefixtures("any_app")
def test_legend_visibility() -> None:
    """Legend entries match channel visibility and names."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15])
    edges = np.array([0, 33, 66, 100], dtype=float)

    hist.set_channel_data(0, counts, edges)
    hist.set_channel_name(0, "FITC")
    hist.set_channel_data(1, counts, edges)
    hist.set_channel_name(1, "DAPI")

    ch0 = hist._channels[0]
    ch1 = hist._channels[1]
    assert ch0.legend_text.text == "● FITC"
    assert ch1.legend_text.text == "● DAPI"

    hist.set_channel_visible(0, False)
    assert not ch0.legend_text.visible

    hist.set_channel_visible(0, True)
    assert ch0.legend_text.visible
