from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication, QWidget

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
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
        if not isinstance(canvas, QWidget):
            raise TypeError(f"Expected canvas to be QWidget, got {type(canvas)}")

        f = MouseEventFilter(canvas, receiver)
        canvas.installEventFilter(f)
        return lambda: canvas.removeEventFilter(f)

    def process_events(self) -> None:
        """Process events for the application."""
        QApplication.processEvents()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        QTimer.singleShot(msec, Qt.TimerType.PreciseTimer, func)


class MouseEventFilter(QObject):
    def __init__(self, canvas: QWidget, receiver: Mouseable):
        super().__init__()
        self.canvas = canvas
        self.receiver = receiver
        self.active_button = MouseButton.NONE

    def mouse_btn(self, btn: Any) -> MouseButton:
        if btn == Qt.MouseButton.LeftButton:
            return MouseButton.LEFT
        if btn == Qt.MouseButton.RightButton:
            return MouseButton.RIGHT
        if btn == Qt.MouseButton.NoButton:
            return MouseButton.NONE

        raise Exception(f"Qt mouse button {btn} is unknown")

    def set_cursor(self, type: CursorType) -> None:
        self.canvas.setCursor(type.to_qt())

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
        if (
            qevent.type() == qevent.Type.ContextMenu
            and type(obj).__name__ == "CanvasBackendDesktop"
        ):
            return False  # pragma: no cover
        if obj is self.canvas or obj in children:
            if isinstance(qevent, QMouseEvent):
                pos = qevent.pos()
                etype = qevent.type()
                btn = self.mouse_btn(qevent.button())
                if etype == QEvent.Type.MouseMove:
                    mme = MouseMoveEvent(x=pos.x(), y=pos.y(), btn=self.active_button)
                    intercept |= receiver.on_mouse_move(mme)
                    if cursor := receiver.get_cursor(mme):
                        self.set_cursor(cursor)
                    receiver.mouseMoved.emit(mme)
                elif etype == QEvent.Type.MouseButtonPress:
                    self.active_button = btn
                    mpe = MousePressEvent(x=pos.x(), y=pos.y(), btn=self.active_button)
                    intercept |= receiver.on_mouse_press(mpe)
                    receiver.mousePressed.emit(mpe)
                elif etype == QEvent.Type.MouseButtonRelease:
                    mre = MouseReleaseEvent(
                        x=pos.x(), y=pos.y(), btn=self.active_button
                    )
                    self.active_button = MouseButton.NONE
                    intercept |= receiver.on_mouse_release(mre)
                    receiver.mouseReleased.emit(mre)
        return intercept
