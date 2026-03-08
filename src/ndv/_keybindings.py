"""Keybinding definitions for ndv."""

from __future__ import annotations

from enum import Enum, auto

from ndv._types import KeyCode, KeyMod


class Action(Enum):
    STEP_FORWARD = auto()
    STEP_BACKWARD = auto()
    FOCUS_NEXT_AXIS = auto()
    FOCUS_PREV_AXIS = auto()


_DEFAULT_KEYBINDINGS: dict[tuple[KeyCode | str, KeyMod], Action] = {
    (KeyCode.RIGHT, KeyMod.NONE): Action.STEP_FORWARD,
    (KeyCode.LEFT, KeyMod.NONE): Action.STEP_BACKWARD,
    (KeyCode.UP, KeyMod.NONE): Action.FOCUS_PREV_AXIS,
    (KeyCode.DOWN, KeyMod.NONE): Action.FOCUS_NEXT_AXIS,
}
