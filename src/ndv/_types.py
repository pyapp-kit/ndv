"""General model for ndv."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from contextlib import suppress
from enum import Enum, IntFlag, auto
from functools import cache
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast

from pydantic import PlainSerializer, PlainValidator
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QWidget
    from wx import Cursor

    from ndv.views.bases import Viewable


def _maybe_int(val: Any) -> Any:
    # try to convert to int if possible
    with suppress(ValueError, TypeError):
        val = int(float(val))
    return val


def _to_slice(val: Any) -> slice:
    # slices are returned as is
    if isinstance(val, slice):
        if not all(
            isinstance(i, (int, type(None))) for i in (val.start, val.stop, val.step)
        ):
            raise TypeError(f"Slice start/stop/step must all be integers: {val!r}")
        return val
    # single integers are converted to slices starting at that index
    if isinstance(val, int):
        return slice(val, val + 1)
    # sequences are interpreted as arguments to the slice constructor
    if isinstance(val, Sequence):
        return slice(*(int(x) if x is not None else None for x in val))
    raise TypeError(f"Expected int or slice, got {type(val)}")


Slice = Annotated[slice, PlainValidator(_to_slice)]

# An axis key is any hashable object that can be used to index an axis
# In many cases it will be an integer, but for some labeled arrays it may be a string
# or other hashable object.  It is up to the DataWrapper to convert these keys to
# actual integer indices.
AxisKey: TypeAlias = Annotated[
    Hashable, PlainValidator(_maybe_int), PlainSerializer(str, return_type=str)
]
# An channel key is any hashable object that can be used to describe a position along
# an axis. In many cases it will be an integer, but it might also provide a contextual
# label for one or more positions.
ChannelKey: TypeAlias = Annotated[
    Hashable, PlainValidator(_maybe_int), PlainSerializer(str, return_type=str)
]


class MouseButton(IntFlag):
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()
    NONE = auto()


class MouseMoveEvent(NamedTuple):
    """Event emitted when the user moves the cursor."""

    x: float
    y: float
    btn: MouseButton = MouseButton.NONE


class MousePressEvent(NamedTuple):
    """Event emitted when mouse button is pressed."""

    x: float
    y: float
    btn: MouseButton


class MouseReleaseEvent(NamedTuple):
    """Event emitted when mouse button is released."""

    x: float
    y: float
    btn: MouseButton


class CursorType(Enum):
    DEFAULT = "default"
    CROSS = "cross"
    V_ARROW = "v_arrow"
    H_ARROW = "h_arrow"
    ALL_ARROW = "all_arrow"
    BDIAG_ARROW = "bdiag_arrow"
    FDIAG_ARROW = "fdiag_arrow"

    def apply_to(self, widget: Viewable) -> None:
        """Applies the cursor type to the given widget."""
        native = widget.frontend_widget()
        if hasattr(native, "setCursor"):
            cast("QWidget", native).setCursor(self.to_qt())

    def to_qt(self) -> Qt.CursorShape:
        """Converts CursorType to Qt.CursorShape."""
        from qtpy.QtCore import Qt

        return {
            CursorType.DEFAULT: Qt.CursorShape.ArrowCursor,
            CursorType.CROSS: Qt.CursorShape.CrossCursor,
            CursorType.V_ARROW: Qt.CursorShape.SizeVerCursor,
            CursorType.H_ARROW: Qt.CursorShape.SizeHorCursor,
            CursorType.ALL_ARROW: Qt.CursorShape.SizeAllCursor,
            CursorType.BDIAG_ARROW: Qt.CursorShape.SizeBDiagCursor,
            CursorType.FDIAG_ARROW: Qt.CursorShape.SizeFDiagCursor,
        }[self]

    def to_jupyter(self) -> str:
        """Converts CursorType to jupyter cursor strings."""
        return {
            CursorType.DEFAULT: "default",
            CursorType.CROSS: "crosshair",
            CursorType.V_ARROW: "ns-resize",
            CursorType.H_ARROW: "ew-resize",
            CursorType.ALL_ARROW: "move",
            CursorType.BDIAG_ARROW: "nesw-resize",
            CursorType.FDIAG_ARROW: "nwse-resize",
        }[self]

    # Note a new object must be created every time. We should cache it!
    @cache
    def to_wx(self) -> Cursor:
        """Converts CursorType to jupyter cursor strings."""
        from wx import (
            CURSOR_ARROW,
            CURSOR_CROSS,
            CURSOR_SIZENESW,
            CURSOR_SIZENS,
            CURSOR_SIZENWSE,
            CURSOR_SIZEWE,
            CURSOR_SIZING,
            Cursor,
        )

        return {
            CursorType.DEFAULT: Cursor(CURSOR_ARROW),
            CursorType.CROSS: Cursor(CURSOR_CROSS),
            CursorType.V_ARROW: Cursor(CURSOR_SIZENS),
            CursorType.H_ARROW: Cursor(CURSOR_SIZEWE),
            CursorType.ALL_ARROW: Cursor(CURSOR_SIZING),
            CursorType.BDIAG_ARROW: Cursor(CURSOR_SIZENESW),
            CursorType.FDIAG_ARROW: Cursor(CURSOR_SIZENWSE),
        }[self]
