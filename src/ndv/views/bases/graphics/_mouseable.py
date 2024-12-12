from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from psygnal import Signal

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

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
        from qtpy.QtCore import QEvent, QObject
        from qtpy.QtGui import QMouseEvent

        if not isinstance(canvas, QObject):
            raise TypeError(f"Expected canvas to be QObject, got {type(canvas)}")

        class Filter(QObject):
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

                intercept = False
                if obj is canvas or obj in children:
                    if isinstance(qevent, QMouseEvent):
                        pos = qevent.pos()
                        etype = qevent.type()
                        if etype == QEvent.Type.MouseMove:
                            mme = MouseMoveEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_move(mme)
                            receiver.mouseMoved.emit(mme)
                        elif etype == QEvent.Type.MouseButtonPress:
                            mpe = MousePressEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_press(mpe)
                            receiver.mousePressed.emit(mpe)
                        elif etype == QEvent.Type.MouseButtonRelease:
                            mre = MouseReleaseEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_release(mre)
                            receiver.mouseReleased.emit(mre)
                return intercept

        f = Filter()
        canvas.installEventFilter(f)
        return lambda: canvas.removeEventFilter(f)

    elif frontend == GuiFrontend.JUPYTER:
        from jupyter_rfb import RemoteFrameBuffer

        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )

        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = canvas.handle_event

        def handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            etype = ev["event_type"]
            if etype == "pointer_move":
                mme = MouseMoveEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_move(mme)
                receiver.mouseMoved.emit(mme)
            elif etype == "pointer_down":
                mpe = MousePressEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_press(mpe)
                receiver.mousePressed.emit(mpe)
            elif etype == "pointer_up":
                mre = MouseReleaseEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_release(mre)
                receiver.mouseReleased.emit(mre)
            super_handle_event(ev)

        canvas.handle_event = MethodType(handle_event, canvas)
        return lambda: setattr(canvas, "handle_event", super_handle_event)

    elif frontend == GuiFrontend.WX:
        from wx import EVT_LEFT_DOWN, EVT_LEFT_UP, EVT_MOTION, EvtHandler, MouseEvent

        if not isinstance(canvas, EvtHandler):
            raise TypeError(
                f"Expected vispy canvas to be wx EvtHandler, got {type(canvas)}"
            )

        # TIP: event.Skip() can be used to allow the event to propagate to other
        # handlers.

        def on_mouse_move(event: MouseEvent) -> None:
            mme = MouseMoveEvent(x=event.GetX(), y=event.GetY())
            receiver.on_mouse_move(mme)
            receiver.mouseMoved.emit(mme)

        def on_mouse_press(event: MouseEvent) -> None:
            mpe = MousePressEvent(x=event.GetX(), y=event.GetY())
            receiver.on_mouse_press(mpe)
            receiver.mousePressed.emit(mpe)

        def on_mouse_release(event: MouseEvent) -> None:
            mre = MouseReleaseEvent(x=event.GetX(), y=event.GetY())
            receiver.on_mouse_release(mre)
            receiver.mouseReleased.emit(mre)

        canvas.Bind(EVT_MOTION, on_mouse_move)
        canvas.Bind(EVT_LEFT_DOWN, on_mouse_press)
        canvas.Bind(EVT_LEFT_UP, on_mouse_release)

        def _unbind() -> None:
            canvas.Unbind(EVT_MOTION, on_mouse_move)
            canvas.Unbind(EVT_LEFT_DOWN, on_mouse_press)
            canvas.Unbind(EVT_LEFT_UP, on_mouse_release)

        return _unbind

    raise RuntimeError(f"Unsupported frontend: {frontend}")
