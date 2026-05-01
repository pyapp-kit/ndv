"""Keybinding definitions for ndv."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from app_model.types import KeyBinding, KeyCode, SimpleKeyBinding
from scenex.app.events import MouseButton, WheelEvent

if TYPE_CHECKING:
    from scenex.app.events import KeyPressEvent

    from ndv._types import AxisKey
    from ndv.controllers._array_viewer import ArrayViewer


class Action(Enum):
    STEP_FORWARD = auto()
    STEP_BACKWARD = auto()
    FOCUS_NEXT_AXIS = auto()
    FOCUS_PREV_AXIS = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()


_DEFAULT_KEYBINDINGS: dict[KeyBinding, Action] = {
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.RightArrow)]): Action.STEP_FORWARD,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.LeftArrow)]): Action.STEP_BACKWARD,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.UpArrow)]): Action.FOCUS_PREV_AXIS,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.DownArrow)]): Action.FOCUS_NEXT_AXIS,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.Equal)]): Action.ZOOM_IN,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.Equal, shift=True)]): Action.ZOOM_IN,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.NumpadAdd)]): Action.ZOOM_IN,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.Minus)]): Action.ZOOM_OUT,
    KeyBinding(parts=[SimpleKeyBinding(key=KeyCode.NumpadSubtract)]): Action.ZOOM_OUT,
}


def handle_key_press(event: KeyPressEvent, viewer: ArrayViewer) -> None:
    """Dispatch a key press event to the appropriate action."""
    action = _DEFAULT_KEYBINDINGS.get(event.key)
    if action is Action.STEP_FORWARD:
        _step_focused_slider(viewer, 1)
    elif action is Action.STEP_BACKWARD:
        _step_focused_slider(viewer, -1)
    elif action is Action.FOCUS_NEXT_AXIS:
        _cycle_focused_axis(viewer, 1)
    elif action is Action.FOCUS_PREV_AXIS:
        _cycle_focused_axis(viewer, -1)
    elif action in [Action.ZOOM_IN, Action.ZOOM_OUT]:
        size = viewer._canvas._canvas.size
        canvas_pos = (size[0] / 2, size[1] / 2)
        angle_delta = (0, 120) if action is Action.ZOOM_IN else (0, -120)
        mouse_event = WheelEvent(canvas_pos, MouseButton.NONE, angle_delta=angle_delta)
        view = viewer._canvas.view
        view.camera.controller.handle_event(mouse_event, view)


def _steppable_axes(viewer: ArrayViewer) -> list[AxisKey]:
    """Non-visible, non-hidden axes that have sliders."""
    return [
        ax
        for ax in viewer._resolved.data_coords
        if ax not in viewer._resolved.visible_axes
        and ax not in viewer._resolved.hidden_sliders
    ]


def _ensure_focused_axis(viewer: ArrayViewer) -> AxisKey | None:
    """Ensure _focused_slider_axis is valid; default to last steppable."""
    axes = _steppable_axes(viewer)
    if not axes:
        return None
    if viewer._focused_slider_axis not in axes:
        viewer._focused_slider_axis = axes[-1]
    return viewer._focused_slider_axis


def _step_focused_slider(viewer: ArrayViewer, delta: int) -> None:
    axis = _ensure_focused_axis(viewer)
    if axis is None:
        return
    ax = cast("int", axis)
    coords = viewer._resolved.data_coords[ax]
    current = viewer._resolved.current_index.get(ax, 0)
    if isinstance(current, slice):
        return
    new_val = max(0, min(current + delta, len(coords) - 1))
    if new_val != current:
        viewer._display_model.current_index[axis] = new_val


def _cycle_focused_axis(viewer: ArrayViewer, direction: int) -> None:
    axes = _steppable_axes(viewer)
    if not axes:
        return
    current = _ensure_focused_axis(viewer)
    if current is None:
        return
    idx = axes.index(current)
    new_idx = (idx + direction) % len(axes)
    viewer._focused_slider_axis = axes[new_idx]
