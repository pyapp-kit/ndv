from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from qtpy.QtCore import QEvent, QObject
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
    from collections.abc import Container
    from concurrent.futures import Future

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import P, T
    from ndv.views.bases._graphics._mouseable import Mouseable


class QtAppWrap(NDVApp):
    """Provider for PyQt5/PySide2/PyQt6/PySide6."""

    _APP_INSTANCE: ClassVar[Any] = None
    IPY_MAGIC_KEY = "qt"

    def create_app(self) -> Any:
        if (qapp := QApplication.instance()) is None:
            # if we're running in IPython
            # start the %gui qt magic if NDV_IPYTHON_MAGIC!=0
            if not self._maybe_enable_ipython_gui():
                # otherwise create a new QApplication
                # must be stored in a class variable to prevent garbage collection
                QtAppWrap._APP_INSTANCE = qapp = QApplication(sys.argv)
                qapp.setOrganizationName("ndv")
                qapp.setApplicationName("ndv")

        self._install_excepthook()
        return qapp

    def run(self) -> None:
        app = QApplication.instance() or self.create_app()

        for wdg in QApplication.topLevelWidgets():
            wdg.raise_()

        if ipy_shell := self._ipython_shell():
            # if we're already in an IPython session with %gui qt, don't block
            if str(ipy_shell.active_eventloop).startswith("qt"):
                return

        app.exec()

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        from ._main_thread import call_in_main_thread

        return call_in_main_thread(func, *args, **kwargs)

    def array_view_class(self) -> type[ArrayView]:
        from ._array_view import QtArrayView

        return QtArrayView

    def filter_mouse_events(
        self, canvas: Any, receiver: Mouseable
    ) -> Callable[[], None]:
        if not isinstance(canvas, QObject):
            raise TypeError(f"Expected canvas to be QObject, got {type(canvas)}")

        f = MouseEventFilter(canvas, receiver)
        canvas.installEventFilter(f)
        return lambda: canvas.removeEventFilter(f)


class MouseEventFilter(QObject):
    def __init__(self, canvas: QObject, receiver: Mouseable):
        super().__init__()
        self.canvas = canvas
        self.receiver = receiver

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
            children: Container = self.canvas.children()
        except RuntimeError:
            # native is likely dead
            return False

        intercept = False
        receiver = self.receiver
        if obj is self.canvas or obj in children:
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
