from types import MethodType
from typing import TYPE_CHECKING, Callable, cast

import vispy.app
from vispy import scene

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.views.protocols import Mouseable

if TYPE_CHECKING:
    from collections.abc import Container


def filter_mouse_events(
    scene_canvas: scene.SceneCanvas, receiver: Mouseable
) -> Callable[[], None]:
    """Intercept mouse events on `scene_canvas` and forward them to `receiver`.

    Parameters
    ----------
    scene_canvas : scene.SceneCanvas
        The vispy canvas to intercept mouse events from.
    receiver : Mouseable
        The object to forward mouse events to.

    Returns
    -------
    Callable[[], None]
        A function that can be called to remove the event filter.
    """
    app = cast("vispy.app.Application", scene_canvas.app)

    if "qt" in str(app.backend_name).lower():
        from qtpy.QtCore import QEvent, QObject
        from qtpy.QtGui import QMouseEvent

        if not isinstance((native := scene_canvas.native), QObject):
            raise TypeError(f"Expected vispy canvas to be QObject, got {type(native)}")

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
                    children: Container = native.children()
                except RuntimeError:
                    # native is likely dead
                    return False

                intercept = False
                if obj is native or obj in children:
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
        native.installEventFilter(f)
        return lambda: native.removeEventFilter(f)

    elif "jupyter" in str(app.backend_name).lower():
        from vispy.app.backends._jupyter_rfb import CanvasBackend

        if not isinstance((native := scene_canvas.native), CanvasBackend):
            raise TypeError(f"Expected vispy canvas to be QObject, got {type(native)}")

        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = native.handle_event

        def handle_event(self: CanvasBackend, ev: dict) -> None:
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

        native.handle_event = MethodType(handle_event, native)
        return lambda: setattr(native, "handle_event", super_handle_event)

    raise NotImplementedError(f"Unsupported backend: {app.backend_name}")
