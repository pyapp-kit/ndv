from __future__ import annotations

import numpy as np
import pytest
from pygfx.objects import WheelEvent

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._lut_model import ClimsManual, LUTModel
from ndv.views._pygfx._histogram import PyGFXHistogramCanvas


@pytest.mark.usefixtures("any_app")
def test_hscroll() -> None:
    model = LUTModel(
        visible=True,
        cmap="red",
        # gamma=2,
    )
    histogram = PyGFXHistogramCanvas()
    histogram.set_range(x=(0, 10), y=(0, 1))
    histogram.model = model
    left, right = 0, 10
    histogram.set_clims((left, right))

    old_x = histogram._camera.local.position[0]
    old_width = histogram._camera.width
    evt = WheelEvent(type="wheel", x=5, y=5, dx=-120, dy=0)
    histogram._controller.handle_event(evt, histogram._plot_view)
    new_x = histogram._camera.local.position[0]
    new_width = histogram._camera.width
    assert new_x < old_x
    assert abs(new_width - old_width) <= 1e-6

    evt = WheelEvent(type="wheel", x=5, y=5, dx=120, dy=0)
    histogram._controller.handle_event(evt, histogram._plot_view)
    new_x = histogram._camera.local.position[0]
    new_width = histogram._camera.width
    assert abs(new_x - old_x) <= 1e-6
    assert abs(new_width - old_width) <= 1e-6

    histogram.close()


@pytest.mark.usefixtures("any_app")
def test_highlight() -> None:
    # Set up a histogram
    histogram = PyGFXHistogramCanvas()
    assert not histogram._highlight.visible
    assert histogram._highlight.local.x == 0
    assert histogram._highlight.local.scale_y == 1

    # Add some data...
    values = np.random.randint(0, 100, (100))
    bin_edges = np.linspace(0, 10, values.size + 1)
    histogram.set_data(values, bin_edges)
    # ...and ensure the scale is updated
    assert histogram._highlight.local.scale_y == values.max() / 0.98

    # Highlight a value...
    histogram.highlight(5)
    # ...and ensure the highlight is shown in the right place
    assert histogram._highlight.visible
    assert histogram._highlight.local.x == 5

    # Remove the highlight...
    histogram.highlight(None)
    # ...and ensure the highlight is hidden
    assert not histogram._highlight.visible

    histogram.close()


@pytest.mark.usefixtures("any_app")
def test_interaction() -> None:
    """Checks basic histogram functionality."""
    model = LUTModel(
        visible=True,
        cmap="red",
        # gamma=2,
    )
    histogram = PyGFXHistogramCanvas()
    histogram.set_range(x=(0, 10), y=(0, 1))
    histogram.model = model
    left, right = 0, 10
    histogram.set_clims((left, right))

    def world_to_canvas(x: float, y: float) -> tuple[float, float]:
        return histogram.world_to_canvas((x, y, 0))

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

    histogram.close()
