"""Keybinding definitions for ndv."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from ndv._types import KeyCode, KeyMod, KeyPressEvent

if TYPE_CHECKING:
    from ndv._types import AxisKey
    from ndv.controllers._array_viewer import ArrayViewer


class Action(Enum):
    STEP_FORWARD = auto()
    STEP_BACKWARD = auto()
    FOCUS_NEXT_AXIS = auto()
    FOCUS_PREV_AXIS = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()


_DEFAULT_KEYBINDINGS: dict[tuple[KeyCode | str, KeyMod], Action] = {
    (KeyCode.RIGHT, KeyMod.NONE): Action.STEP_FORWARD,
    (KeyCode.LEFT, KeyMod.NONE): Action.STEP_BACKWARD,
    (KeyCode.UP, KeyMod.NONE): Action.FOCUS_PREV_AXIS,
    (KeyCode.DOWN, KeyMod.NONE): Action.FOCUS_NEXT_AXIS,
    ("+", KeyMod.SHIFT): Action.ZOOM_IN,
    ("=", KeyMod.NONE): Action.ZOOM_IN,
    ("-", KeyMod.NONE): Action.ZOOM_OUT,
    ("_", KeyMod.SHIFT): Action.ZOOM_OUT,
}


def handle_key_press(event: KeyPressEvent, viewer: ArrayViewer) -> None:
    """Dispatch a key press event to the appropriate action."""
    action = _DEFAULT_KEYBINDINGS.get((event.key, event.mods))
    if action is Action.STEP_FORWARD:
        _step_focused_slider(viewer, 1)
    elif action is Action.STEP_BACKWARD:
        _step_focused_slider(viewer, -1)
    elif action is Action.FOCUS_NEXT_AXIS:
        _cycle_focused_axis(viewer, 1)
    elif action is Action.FOCUS_PREV_AXIS:
        _cycle_focused_axis(viewer, -1)
    elif action is Action.ZOOM_IN:
        viewer._canvas.zoom(factor=0.667, center=viewer._mouse_canvas_pos)
    elif action is Action.ZOOM_OUT:
        viewer._canvas.zoom(factor=1.5, center=viewer._mouse_canvas_pos)


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
