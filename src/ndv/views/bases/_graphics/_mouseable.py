from __future__ import annotations

from psygnal import Signal

from ndv._types import CursorType, MouseMoveEvent, MousePressEvent, MouseReleaseEvent


class Mouseable:
    """Mixin class for objects that can be interacted with using the mouse.

    The signals here are to be emitted by the view object that inherits this class;
    usually by intercepting native mouse events with `filter_mouse_events`.

    The methods allow the object to handle its own mouse events before emitting the
    signals. If the method returns `True`, the event is considered handled and should
    not be passed to the next receiver in the chain.
    """

    mouseMoved = Signal(MouseMoveEvent)
    mousePressed = Signal(MousePressEvent)
    mouseReleased = Signal(MouseReleaseEvent)

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False

    def get_cursor(self, event: MouseMoveEvent) -> CursorType | None:
        return None
