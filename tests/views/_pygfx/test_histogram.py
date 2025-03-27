from __future__ import annotations

import pytest
from pytest import fixture

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._lut_model import ClimsManual, LUTModel
from ndv.views._pygfx._histogram import PyGFXHistogramCanvas


@fixture
def model() -> LUTModel:
    return LUTModel(
        visible=True,
        cmap="red",
        # gamma=2,
    )


@fixture
def histogram() -> PyGFXHistogramCanvas:
    canvas = PyGFXHistogramCanvas()
    canvas.set_range(x=(0, 10), y=(0, 1))
    return canvas


# FIXME: These leaks are very consistent
@pytest.mark.allow_leaks
@pytest.mark.usefixtures("any_app")
def test_interaction(model: LUTModel, histogram: PyGFXHistogramCanvas) -> None:
    """Checks basic histogram functionality."""
    histogram.model = model
    left, right = 0, 10
    histogram.set_clims((left, right))

    def world_to_canvas(x: float, y: float) -> tuple[float, float]:
        return histogram.world_to_canvas((x, y, 0))

    # TODO: Uncomment code with https://github.com/pyapp-kit/ndv/pull/158
    # Test cursors
    # x, y = world_to_canvas((left + right) / 2, 0.5)
    # assert (
    #     histogram.get_cursor(MouseMoveEvent(x=x, y=y, btn=MouseButton.NONE))
    #     == CursorType.V_ARROW
    # )
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
    # x, y = world_to_canvas((left + right) / 2, 0.5)
    # histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    # x, y = world_to_canvas((left + right) / 2, 0.75)
    # histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    # histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    # assert model.gamma == -np.log2(0.75)

    # Double clicking gamma resets to 1.
    # x, y = world_to_canvas((left + right) / 2, 0.75)
    # histogram.on_mouse_double_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    # assert model.gamma == 1

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
