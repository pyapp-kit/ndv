"""Tests SharedHistogram visual behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from ndv.views._shared_histogram import SharedHistogram

if TYPE_CHECKING:
    import scenex as snx


@fixture
def hist() -> SharedHistogram:
    canvas = SharedHistogram()
    canvas.set_range(x=(0, 100), y=(0, 1))
    return canvas


# def _world_to_canvas(
#     hist: SharedHistogram, x: float, y: float
# ) -> tuple[float, float]:
#     cam = hist.view.camera
#     return cam.
#     return tuple(hist.node_tform.imap((x, y))[:2])  # type: ignore[return-value]


# ---------- Channel lifecycle ----------


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


# ---------- Visibility ----------


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


# ---------- key=None (grayscale channel) ----------


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


# ---------- Clim/gamma interaction ----------


def _world_to_canvas(view: snx.View, x: float, y: float) -> tuple[float, float]:
    return view.camera.projection.map(view.camera.transform.imap((x, y, 0)))[:2]  # type: ignore[return-value]


@pytest.mark.usefixtures("any_app")
def test_clim_drag_emits_signal() -> None:
    """Dragging a clim handle emits climsChanged with correct key."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(0, 100))  # ensure we have a known canvas width
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (0, 1, 0, 1))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Grab the left clim
    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    left_clim_pos = (
        x_start + (width / 10),
        y_start + height - 1,
    )
    hist.canvas.handle(MousePressEvent(pos=left_clim_pos, buttons=MouseButton.LEFT))
    # Drag to new position
    new_left_clim_pos = (
        x_start + (width * 3 / 10),
        y_start + height - 1,
    )
    hist.canvas.handle(MouseMoveEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT)
    )

    assert len(received) >= 1
    assert received[-1][0] == 0  # correct channel key
    assert received[-1][1][0] >= 25  # left clim moved right (approx)


@pytest.mark.usefixtures("any_app")
def test_none_key_clim_drag() -> None:
    """Clim dragging works for key=None (grayscale channel)."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(None, counts, edges)
    hist.set_range(x=(0, 100))  # ensure we have a known canvas width
    hist.set_channel_clims(None, (10, 90))
    hist.set_channel_color(None, (0.5, 0.5, 0.5, 1))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Grab the right clim
    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    right_clim_pos = (
        x_start + (width * 9 / 10),
        y_start + height - 1,
    )
    hist.canvas.handle(MousePressEvent(pos=right_clim_pos, buttons=MouseButton.LEFT))
    new_right_clim_pos = (
        x_start + (width * 7 / 10),
        y_start + height - 1,
    )
    hist.canvas.handle(MouseMoveEvent(pos=new_right_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_right_clim_pos, buttons=MouseButton.LEFT)
    )

    assert len(received) >= 1
    assert received[-1][0] is None  # key is None, not _NO_KEY


@pytest.mark.usefixtures("any_app")
def test_gamma_double_click_resets() -> None:
    """Double-clicking gamma handle emits gammaChanged with 1.0."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(0, 100))  # ensure we have a known canvas width
    hist.set_channel_clims(0, (0, 100))
    hist.set_channel_gamma(0, 2.0)
    hist.set_channel_color(0, (0, 1, 0, 1))

    received: list = []
    hist.gammaChanged.connect(lambda key, gamma: received.append((key, gamma)))

    # Find gamma handle position: midpoint of clims, y = 2^(-gamma) * y_top
    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    gamma_height = 2 ** (-2.0)
    gamma_pos = (x_start + (width / 2), y_start + height * (1 - gamma_height))
    hist.canvas.handle(MouseDoublePressEvent(pos=gamma_pos, buttons=MouseButton.LEFT))

    assert len(received) == 1
    assert received[0] == (0, 1.0)


# ---------- Clim bounds ----------


@pytest.mark.usefixtures("any_app")
def test_clim_bounds_constrain_drag() -> None:
    """Clim drag respects clim_bounds."""
    hist = SharedHistogram()
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_range(x=(-50, 150))  # ensure we have a known canvas width
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (1, 0, 0, 1))
    hist.set_clim_bounds((0, 255))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Try to drag left clim below 0
    # Grab the left clim
    x_start, y_start, width, height = hist.canvas.content_rect_for(hist.view)
    left_clim_pos = (
        x_start + (6 * width / 20),
        y_start + height - 1,
    )
    hist.canvas.handle(MousePressEvent(pos=left_clim_pos, buttons=MouseButton.LEFT))
    new_left_clim_pos = (
        x_start + (2 * width / 20),
        y_start + height - 1,
    )
    hist.canvas.handle(MouseMoveEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT))
    hist.canvas.handle(
        MouseReleaseEvent(pos=new_left_clim_pos, buttons=MouseButton.LEFT)
    )

    if received:
        # Left clim should be clamped to 0, not -50
        assert received[-1][1][0] >= 0
    hist.climsChanged.disconnect()


# ---------- Log scale ----------


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


# ---------- Highlight ----------


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


# ---------- Legend ----------


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

    # Hiding a channel hides its legend
    hist.set_channel_visible(0, False)
    assert not ch0.legend_text.visible

    hist.set_channel_visible(0, True)
    assert ch0.legend_text.visible
