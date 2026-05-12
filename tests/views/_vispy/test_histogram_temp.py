from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pytest import fixture
from scenex.app import CursorType
from scenex.app.events import (
    MouseButton,
    MouseDoublePressEvent,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
    WheelEvent,
)

from ndv.models._lut_model import ClimsManual, LUTModel
from ndv.views._histogram import Histogram


@fixture
def model() -> LUTModel:
    return LUTModel(
        visible=True,
        cmap="red",
        gamma=1,
    )


@fixture
def histogram() -> Histogram:
    # Set up a histogram
    histogram = Histogram()

    # Add some data...
    values = np.random.randint(0, 100, (100))
    bin_edges = np.linspace(0, 10, values.size + 1)
    histogram.set_data(values, bin_edges)
    histogram.set_range(x=(-2, 12), y=(0, 1))
    return histogram


@pytest.mark.usefixtures("any_app")
def test_hscroll(histogram: Histogram) -> None:
    cam = histogram.view.camera

    def get_extents() -> tuple[float, float]:
        left, *_ = cam.transform.map(cam.projection.imap((-1, 0)))
        right, *_ = cam.transform.map(cam.projection.imap((1, 0)))
        return float(left), float(right)

    old_left, old_right = get_extents()
    old_width = old_right - old_left

    x, y, w, h = histogram.canvas.content_rect_for(histogram.view)
    center = (x + w / 2, y + h / 2)

    histogram.canvas.handle(
        WheelEvent(pos=center, buttons=MouseButton.NONE, angle_delta=(1, 0))
    )
    new_left, new_right = get_extents()
    assert new_left < old_left
    assert abs((new_right - new_left) - old_width) <= 1e-6

    histogram.canvas.handle(
        WheelEvent(pos=center, buttons=MouseButton.NONE, angle_delta=(-1, 0))
    )
    new_left, new_right = get_extents()
    assert abs(new_left - old_left) <= 1e-6
    assert abs((new_right - new_left) - old_width) <= 1e-6


@pytest.mark.usefixtures("any_app")
def test_highlight(histogram: Histogram) -> None:
    # Ensure the line is present
    line = histogram.highlight_line
    assert line is not None
    assert not line.visible

    # Highlight a value...
    histogram.highlight(5)
    # ...and ensure the highlight is shown in the right place
    assert line.visible
    assert 5 == line.transform.root[3, 0]

    # Remove the highlight...
    histogram.highlight(None)
    # ...and ensure the highlight is hidden
    assert not line.visible


@pytest.mark.usefixtures("any_app")
def test_interaction(model: LUTModel, histogram: Histogram) -> None:
    """Checks basic histogram functionality."""
    histogram.model = model
    left, right = 0, 10
    histogram.set_clims((left, right))

    def world_to_canvas(x: float, y: float) -> tuple[float, float]:
        cam = histogram.view.camera
        view_pos_ndc = cam.projection.map(cam.transform.imap((x, y, 0)))
        x, y, w, h = histogram.canvas.content_rect_for(histogram.view)
        view_pos = (view_pos_ndc[0] * w / 2 + w / 2, -view_pos_ndc[1] * h / 2 + h / 2)
        return (x + view_pos[0], y + view_pos[1])

    # Test cursors
    pos = world_to_canvas((left + right) / 2, 0.5)
    with patch("ndv.views._histogram.snx.set_cursor") as mock_set_cursor:
        histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.NONE))
    mock_set_cursor.assert_called_once_with(histogram.canvas, CursorType.V_ARROW)

    pos = world_to_canvas(left, 0.5)
    with patch("ndv.views._histogram.snx.set_cursor") as mock_set_cursor:
        histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.NONE))
    mock_set_cursor.assert_called_once_with(histogram.canvas, CursorType.H_ARROW)
    pos = world_to_canvas(right, 0.5)
    with patch("ndv.views._histogram.snx.set_cursor") as mock_set_cursor:
        histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.NONE))
    mock_set_cursor.assert_called_once_with(histogram.canvas, CursorType.H_ARROW)

    # Select and move gamma
    pos = world_to_canvas((left + right) / 2, 0.5)
    histogram.canvas.handle(MousePressEvent(pos=pos, buttons=MouseButton.LEFT))
    pos = world_to_canvas((left + right) / 2, 0.75)
    histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.LEFT))
    histogram.canvas.handle(MouseReleaseEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.gamma == -np.log2(0.75)

    # Double clicking gamma resets to 1.
    pos = world_to_canvas((left + right) / 2, 0.75)
    histogram.canvas.handle(MouseDoublePressEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.gamma == 1

    # Select and move the left clim
    pos = world_to_canvas(left, 0.5)
    histogram.canvas.handle(MousePressEvent(pos=pos, buttons=MouseButton.LEFT))
    left = 1
    pos = world_to_canvas(left, 0.5)
    histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.LEFT))
    histogram.canvas.handle(MouseReleaseEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)

    # Select and move the right clim
    pos = world_to_canvas(right, 0.5)
    histogram.canvas.handle(MousePressEvent(pos=pos, buttons=MouseButton.LEFT))
    right = 9
    pos = world_to_canvas(right, 0.5)
    histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.LEFT))
    histogram.canvas.handle(MouseReleaseEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)

    # Ensure the right clim cannot move beyond the left clim
    pos = world_to_canvas(right, 0.5)
    histogram.canvas.handle(MousePressEvent(pos=pos, buttons=MouseButton.LEFT))
    right = 0
    pos = world_to_canvas(right, 0.5)
    histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.LEFT))
    histogram.canvas.handle(MouseReleaseEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=left)

    # Ensure right clim is chosen when overlapping
    pos = world_to_canvas(left, 0.5)
    histogram.canvas.handle(MousePressEvent(pos=pos, buttons=MouseButton.LEFT))
    right = 9
    pos = world_to_canvas(right, 0.5)
    histogram.canvas.handle(MouseMoveEvent(pos=pos, buttons=MouseButton.LEFT))
    histogram.canvas.handle(MouseReleaseEvent(pos=pos, buttons=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)
