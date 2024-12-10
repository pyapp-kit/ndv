from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Callable, cast

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

if TYPE_CHECKING:
    from collections.abc import Container

    from wgpu.gui.jupyter import JupyterWgpuCanvas
    from wgpu.gui.qt import QWgpuCanvas

    from ndv.views.protocols import Mouseable


def filter_mouse_events(
    wgpu_canvas: QWgpuCanvas | JupyterWgpuCanvas, receiver: Mouseable
) -> Callable[[], None]:
    """Intercept mouse events on `scene_canvas` and forward them to `receiver`.

    Parameters
    ----------
    wgpu_canvas : scene.SceneCanvas
        The vispy canvas to intercept mouse events from.
    receiver : Mouseable
        The object to forward mouse events to.

    Returns
    -------
    Callable[[], None]
        A function that can be called to remove the event filter.
    """
    type_name = str(type(wgpu_canvas))
    if "QWgpu" in type_name:
        from qtpy.QtCore import QEvent, QObject
        from qtpy.QtGui import QMouseEvent

        if not isinstance(wgpu_canvas, QObject):
            raise TypeError(
                f"Expected vispy canvas to be QObject, got {type(wgpu_canvas)}"
            )

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
                    children: Container = wgpu_canvas.children()
                except RuntimeError:
                    # wgpu_canvas is likely dead
                    return False

                intercept = False
                if obj is wgpu_canvas or obj in children:
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
        wgpu_canvas.installEventFilter(f)
        return lambda: wgpu_canvas.removeEventFilter(f)

    elif "Jupyter" in type_name:
        from wgpu.gui.jupyter import RemoteFrameBuffer

        if not isinstance((wgpu_canvas := wgpu_canvas), RemoteFrameBuffer):
            raise TypeError(
                f"Expected vispy canvas to be QObject, got {type(wgpu_canvas)}"
            )
        cast("RemoteFrameBuffer", wgpu_canvas)
        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = wgpu_canvas.handle_event

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

        wgpu_canvas.handle_event = MethodType(handle_event, wgpu_canvas)
        return lambda: setattr(wgpu_canvas, "handle_event", super_handle_event)

    raise NotImplementedError(f"Unsupported canvas type: {wgpu_canvas}")
