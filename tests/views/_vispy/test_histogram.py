from __future__ import annotations

from pytest import fixture

from ndv._types import MouseButton, MouseMoveEvent, MousePressEvent, MouseReleaseEvent
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
    canvas.set_range(x=(0, 100), y=(0, 1))
    return canvas


def test_interaction(model: LUTModel, histogram: VispyHistogramCanvas) -> None:
    """Checks basic histogram functionality."""
    # Setup
    histogram.model = model
    left, right = 20, 80
    histogram.set_clims((left, right))

    def world_to_canvas(x: float, y: float) -> tuple[float, float]:
        return tuple(histogram.node_tform.imap((x, y))[:2])

    # Select and move the left clim
    x, y = world_to_canvas(left, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    left = 30
    x, y = world_to_canvas(left, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)

    # Select and move the right clim
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_press(MousePressEvent(x=x, y=y, btn=MouseButton.LEFT))
    right = 50
    x, y = world_to_canvas(right, 0)
    histogram.on_mouse_move(MouseMoveEvent(x=x, y=y, btn=MouseButton.LEFT))
    histogram.on_mouse_release(MouseReleaseEvent(x=x, y=y, btn=MouseButton.LEFT))
    assert model.clims == ClimsManual(min=left, max=right)
