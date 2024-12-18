from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from psygnal import Signal

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)

if TYPE_CHECKING:
    from collections.abc import Container


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

    def get_cursor(self, canvas_pos: tuple[float, float]) -> CursorType | None:
        return None


def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]:
    """Intercept mouse events on `scene_canvas` and forward them to `receiver`.

    Parameters
    ----------
    canvas : Any
        The front-end canvas widget to intercept mouse events from.
    receiver : Mouseable
        The object to forward mouse events to.

    Returns
    -------
    Callable[[], None]
        A function that can be called to remove the event filter.
    """
    from ndv.views._app import GuiFrontend, gui_frontend

    frontend = gui_frontend()
    if frontend == GuiFrontend.QT:
        from qtpy.QtCore import QEvent, QObject, Qt
        from qtpy.QtGui import QMouseEvent

        if not isinstance(canvas, QObject):
            raise TypeError(f"Expected vispy canvas to be QObject, got {type(canvas)}")

        class Filter(QObject):
            pressed: MouseButton = MouseButton.NONE

            def eventFilter(self, obj: QObject | None, qevent: QEvent | None) -> bool:
                """Event filter installed on the canvas to handle mouse events.

                here is where we get a chance to intercept mouse events before allowing
                the canvas to respond to them. Return `True` to prevent the event from
                being passed to the canvas.
                """
                if qevent is None:
                    return False  # pragma: no cover

                try:
                    # use children in case backend has a subwidget stealing events.
                    children: Container = canvas.children()
                except RuntimeError:
                    # native is likely dead
                    return False
                # FIXME

                intercept = False
                if obj is canvas or obj in children:
                    if isinstance(qevent, QMouseEvent):
                        pos = qevent.pos()
                        etype = qevent.type()
                        if etype == QEvent.Type.MouseMove:
                            mme = MouseMoveEvent(x=pos.x(), y=pos.y(), btn=self.pressed)
                            intercept |= receiver.on_mouse_move(mme)
                            receiver.mouseMoved.emit(mme)
                        elif etype == QEvent.Type.MouseButtonPress:
                            qbtn = qevent.button()
                            self.pressed = (
                                MouseButton.LEFT
                                if qbtn == Qt.MouseButton.LeftButton
                                else MouseButton.RIGHT
                                if qbtn == Qt.MouseButton.RightButton
                                else MouseButton.MIDDLE
                                if qbtn == Qt.MouseButton.MiddleButton
                                else MouseButton.NONE
                            )
                            mpe = MousePressEvent(
                                x=pos.x(), y=pos.y(), btn=self.pressed
                            )
                            intercept |= receiver.on_mouse_press(mpe)
                            receiver.mousePressed.emit(mpe)
                        elif etype == QEvent.Type.MouseButtonRelease:
                            mre = MouseReleaseEvent(
                                x=pos.x(), y=pos.y(), btn=self.pressed
                            )
                            self.pressed = MouseButton.NONE
                            intercept |= receiver.on_mouse_release(mre)
                            receiver.mouseReleased.emit(mre)
                        # FIXME: Ugly
                        if obj and hasattr(obj, "setCursor"):
                            if cursor := receiver.get_cursor((pos.x(), pos.y())):
                                obj.setCursor(cursor.to_qt())
                return intercept

        qfilter = Filter()
        canvas.installEventFilter(qfilter)
        return lambda: canvas.removeEventFilter(qfilter)

    elif frontend == GuiFrontend.JUPYTER:
        from jupyter_rfb import RemoteFrameBuffer

        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(f"Expected vispy canvas to be QObject, got {type(canvas)}")

        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = canvas.handle_event

        # NB A closure is used here to retain state of current button press.
        class JupyterFilter:
            pressed = MouseButton.NONE

            def handle(self, rfb: RemoteFrameBuffer, ev: dict) -> None:
                intercepted = False
                etype = ev["event_type"]
                if etype in ["pointer_move", "pointer_down", "pointer_up"]:
                    btn = ev.get("button", 3)
                    x, y = ev["x"], ev["y"]
                    if etype == "pointer_move":
                        mme = MouseMoveEvent(x=x, y=y, btn=self.pressed)
                        intercepted |= receiver.on_mouse_move(mme)
                        receiver.mouseMoved.emit(mme)
                    elif etype == "pointer_down":
                        self.pressed = (
                            MouseButton.LEFT
                            if btn == 1
                            else MouseButton.RIGHT
                            if btn == 2
                            else MouseButton.MIDDLE
                            if btn == 3
                            else MouseButton.NONE
                        )
                        mpe = MousePressEvent(x=x, y=y, btn=self.pressed)
                        intercepted |= receiver.on_mouse_press(mpe)
                        receiver.mousePressed.emit(mpe)
                    elif etype == "pointer_up":
                        mre = MouseReleaseEvent(x=x, y=y, btn=self.pressed)
                        self.pressed = MouseButton.NONE
                        intercepted |= receiver.on_mouse_release(mre)
                        receiver.mouseReleased.emit(mre)

                    if cursor := receiver.get_cursor((x, y)):
                        canvas.cursor = cursor.to_jupyter()

                if not intercepted:
                    super_handle_event(ev)

        jfilter = JupyterFilter()
        canvas.handle_event = MethodType(jfilter.handle, canvas)
        return lambda: setattr(canvas, "handle_event", super_handle_event)

    raise NotImplementedError(f"Unsupported frontend for mouse events: {frontend!r}")
