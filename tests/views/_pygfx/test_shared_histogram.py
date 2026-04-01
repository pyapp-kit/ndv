"""Tests for PyGFXSharedHistogramCanvas visual behavior."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import fixture

from ndv._types import (
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.views._pygfx._shared_histogram import PyGFXSharedHistogramCanvas


def _force_canvas_size(
    canvas: PyGFXSharedHistogramCanvas, w: int = 600, h: int = 600
) -> None:
    """Force the rendercanvas to report a valid size (needed before show)."""
    rc = canvas._canvas
    rc._size_info.set_physical_size(w, h, 1.0)


@fixture
def hist() -> PyGFXSharedHistogramCanvas:
    canvas = PyGFXSharedHistogramCanvas()
    _force_canvas_size(canvas)
    canvas.set_range(x=(0, 100), y=(0, 1))
    return canvas


def _world_to_canvas(
    hist: PyGFXSharedHistogramCanvas, x: float, y: float
) -> tuple[float, float]:
    return hist.world_to_canvas((x, y, 0))


# ---------- Channel lifecycle ----------


@pytest.mark.usefixtures("any_app")
def test_channel_creation(hist: PyGFXSharedHistogramCanvas) -> None:
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
def test_channel_removal(hist: PyGFXSharedHistogramCanvas) -> None:
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
def test_channel_visibility(hist: PyGFXSharedHistogramCanvas) -> None:
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
    hist = PyGFXSharedHistogramCanvas()
    _force_canvas_size(hist)
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


@pytest.mark.usefixtures("any_app")
def test_clim_drag_emits_signal() -> None:
    """Dragging a clim handle emits climsChanged with correct key."""
    hist = PyGFXSharedHistogramCanvas()
    _force_canvas_size(hist)
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (0, 1, 0, 1))
    hist.set_range(x=(0, 100))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Grab the left clim
    lx, ly = _world_to_canvas(hist, 10, 0)
    hist.on_mouse_press(MousePressEvent(x=lx, y=ly, btn=MouseButton.LEFT))
    # Drag to new position
    nx, ny = _world_to_canvas(hist, 30, 0)
    hist.on_mouse_move(MouseMoveEvent(x=nx, y=ny, btn=MouseButton.LEFT))
    hist.on_mouse_release(MouseReleaseEvent(x=nx, y=ny, btn=MouseButton.LEFT))

    assert len(received) >= 1
    assert received[-1][0] == 0  # correct channel key
    assert received[-1][1][0] >= 25  # left clim moved right (approx)


@pytest.mark.usefixtures("any_app")
def test_none_key_clim_drag() -> None:
    """Clim dragging works for key=None (grayscale channel)."""
    hist = PyGFXSharedHistogramCanvas()
    _force_canvas_size(hist)
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(None, counts, edges)
    hist.set_channel_clims(None, (10, 90))
    hist.set_channel_color(None, (0.5, 0.5, 0.5, 1))
    hist.set_range(x=(0, 100))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Grab the right clim
    rx, ry = _world_to_canvas(hist, 90, 0)
    hist.on_mouse_press(MousePressEvent(x=rx, y=ry, btn=MouseButton.LEFT))
    nx, ny = _world_to_canvas(hist, 70, 0)
    hist.on_mouse_move(MouseMoveEvent(x=nx, y=ny, btn=MouseButton.LEFT))
    hist.on_mouse_release(MouseReleaseEvent(x=nx, y=ny, btn=MouseButton.LEFT))

    assert len(received) >= 1
    assert received[-1][0] is None  # key is None, not _NO_KEY


@pytest.mark.usefixtures("any_app")
def test_gamma_double_click_resets() -> None:
    """Double-clicking gamma handle emits gammaChanged with 1.0."""
    hist = PyGFXSharedHistogramCanvas()
    _force_canvas_size(hist)
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_channel_clims(0, (0, 100))
    hist.set_channel_gamma(0, 2.0)
    hist.set_channel_color(0, (0, 1, 0, 1))
    hist.set_range(x=(0, 100))

    received: list = []
    hist.gammaChanged.connect(lambda key, gamma: received.append((key, gamma)))

    # Find gamma handle position: midpoint of clims, y = 2^(-gamma) * y_top
    y_range = hist._compute_y_range()
    y_top = (y_range[1] if y_range else 1.0) * 0.98
    mid_x = 50.0
    mid_y = (2 ** (-2.0)) * y_top
    gx, gy = _world_to_canvas(hist, mid_x, mid_y)
    hist.on_mouse_double_press(MousePressEvent(x=gx, y=gy, btn=MouseButton.LEFT))

    assert len(received) == 1
    assert received[0] == (0, 1.0)


# ---------- Clim bounds ----------


@pytest.mark.usefixtures("any_app")
def test_clim_bounds_constrain_drag() -> None:
    """Clim drag respects clim_bounds."""
    hist = PyGFXSharedHistogramCanvas()
    _force_canvas_size(hist)
    counts = np.array([5, 10, 15, 10, 5])
    edges = np.linspace(0, 100, 6)
    hist.set_channel_data(0, counts, edges)
    hist.set_channel_clims(0, (10, 90))
    hist.set_channel_color(0, (1, 0, 0, 1))
    hist.set_clim_bounds((0, 255))
    hist.set_range(x=(0, 100))

    received: list = []
    hist.climsChanged.connect(lambda key, clims: received.append((key, clims)))

    # Try to drag left clim below 0
    lx, ly = _world_to_canvas(hist, 10, 0)
    hist.on_mouse_press(MousePressEvent(x=lx, y=ly, btn=MouseButton.LEFT))
    nx, ny = _world_to_canvas(hist, -50, 0)
    hist.on_mouse_move(MouseMoveEvent(x=nx, y=ny, btn=MouseButton.LEFT))
    hist.on_mouse_release(MouseReleaseEvent(x=nx, y=ny, btn=MouseButton.LEFT))

    if received:
        # Left clim should be clamped to 0, not -50
        assert received[-1][1][0] >= 0


# ---------- Log scale ----------


@pytest.mark.usefixtures("any_app")
def test_log_scale() -> None:
    """Log scale can be toggled without errors."""
    hist = PyGFXSharedHistogramCanvas()
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
    hist = PyGFXSharedHistogramCanvas()
    assert not hist._highlight_lines

    hist.highlight({"ch0": 50})
    assert hist._highlight_lines["ch0"].visible

    hist.highlight({})
    assert not hist._highlight_lines["ch0"].visible


# ---------- Legend ----------


@pytest.mark.usefixtures("any_app")
def test_legend_names() -> None:
    """Legend entries track channel names."""
    hist = PyGFXSharedHistogramCanvas()
    counts = np.array([5, 10, 15])
    edges = np.array([0, 33, 66, 100], dtype=float)

    hist.set_channel_data(0, counts, edges)
    hist.set_channel_name(0, "FITC")
    hist.set_channel_data(1, counts, edges)
    hist.set_channel_name(1, "DAPI")

    ch0 = hist._channels[0]
    ch1 = hist._channels[1]
    assert ch0.name == "FITC"
    assert ch1.name == "DAPI"

    # Hiding a channel updates state
    hist.set_channel_visible(0, False)
    assert ch0.visible is False

    hist.set_channel_visible(0, True)
    assert ch0.visible is True
