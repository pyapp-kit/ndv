from __future__ import annotations

import numpy as np
import pytest
from pytest import fixture
from vispy.app.canvas import MouseEvent
from vispy.scene.events import SceneMouseEvent

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._lut_model import ClimsManual, LUTModel
from ndv.views._vispy._histogram import VispyHistogramCanvas


@fixture
def model() -> LUTModel:
    return LUTModel(
        visible=True,
        cmap="red",
        gamma=2,
    )


@fixture
def histogram() -> VispyHistogramCanvas:
    canvas = VispyHistogramCanvas()
    canvas.set_range(x=(0, 10), y=(0, 1))
    return canvas


@pytest.mark.usefixtures("any_app")
def test_hscroll(histogram: VispyHistogramCanvas) -> None:
    old_rect = histogram.plot.camera.rect
    evt = SceneMouseEvent(
        MouseEvent(type="mouse_wheel", delta=[1, 0]), histogram.plot.camera.viewbox
    )
    histogram.plot.camera.viewbox_mouse_event(evt)
    new_rect = histogram.plot.camera.rect
    assert new_rect.left < old_rect.left
    assert abs(new_rect.width - old_rect.width) <= 1e-6

    evt = SceneMouseEvent(
        MouseEvent(type="mouse_wheel", delta=[-1, 0]), histogram.plot.camera.viewbox
    )
    histogram.plot.camera.viewbox_mouse_event(evt)
    new_rect = histogram.plot.camera.rect
    assert abs(new_rect.left - old_rect.left) <= 1e-6
    assert abs(new_rect.width - old_rect.width) <= 1e-6


@pytest.mark.usefixtures("any_app")
def test_highlight() -> None:
    # Set up a histogram
    histogram = VispyHistogramCanvas()
    assert not histogram._highlight.visible
    tform = histogram._highlight.transform
    assert np.allclose(tform.map(histogram._highlight.pos)[:, :2], ((0, 0), (0, 1)))

    # Add some data...
    values = np.random.randint(0, 100, (100))
    bin_edges = np.linspace(0, 10, values.size + 1)
    histogram.set_data(values, bin_edges)
    # ...and ensure the scale is updated
    assert np.allclose(
        tform.map(histogram._highlight.pos)[:, :2], ((0, 0), (0, values.max() / 0.98))
    )

    # Highlight a value...
    histogram.highlight(5)
    # ...and ensure the highlight is shown in the right place
    assert histogram._highlight.visible
    assert np.allclose(
        tform.map(histogram._highlight.pos)[:, :2], ((5, 0), (5, values.max() / 0.98))
    )

    # Remove the highlight...
    histogram.highlight(None)
    # ...and ensure the highlight is hidden
    assert not histogram._highlight.visible

    histogram.close()


@pytest.mark.usefixtures("any_app")
def test_interaction(model: LUTModel, histogram: VispyHistogramCanvas) -> None:
    """Checks basic histogram functionality."""
    histogram.model = model
    left, right = 0, 10
    histogram.set_clims((left, right))

    def world_to_canvas(x: float, y: float) -> tuple[float, float]:
        return tuple(histogram.node_tform.imap((x, y))[:2])

    # Test cursors
    x, y = world_to_canvas((left + right) / 2, 0.5)
    assert (
        histogram.get_cursor(MouseMoveEvent(x=x, y=y, btn=MouseButton.NONE))
        == CursorType.V_ARROW
    )
    x, y = world_to_canvas(left, 0)
    assert (
        histogram.get_cursor(MouseMoveEvent(x=x, y=y, btn=MouseButton.NONE))
        == CursorType.H_ARROW
    )
    x, y = world_to_canvas(right, 0)
    assert (
        histogram.get_cursor(MouseMoveEvent(x=x, y=y, btn=MouseButton.NONE))
        == CursorType.H_ARROW
    )

    # Select and move gamma
    x, y = world_to_canvas((left + right) / 2, 0.5)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    x, y = world_to_canvas((left + right) / 2, 0.75)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.gamma == -np.log2(0.75)

    # Double clicking gamma resets to 1.
    x, y = world_to_canvas((left + right) / 2, 0.75)
    histogram.on_mouse_double_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.gamma == 1

    # Select and move the left clim
    x, y = world_to_canvas(left, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    left = 1
    x, y = world_to_canvas(left, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)

    # Select and move the right clim
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    right = 9
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)

    # Ensure the right clim cannot move beyond the left clim
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    right = 0
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=left)

    # Ensure right clim is chosen when overlapping
    x, y = world_to_canvas(left, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    right = 9
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)
