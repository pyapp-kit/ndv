from typing import TYPE_CHECKING, Any, cast

import vispy.app
from vispy import scene

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.views.protocols import Mouseable

if TYPE_CHECKING:
    from collections.abc import Container


def intercept_mouse_events(scene_canvas: scene.SceneCanvas, receiver: Mouseable) -> Any:
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
        return f
