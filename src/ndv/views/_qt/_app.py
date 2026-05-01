from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import QApplication, QWidget

from ndv._types import (
    KeyCode,
    KeyMod,
    KeyPressEvent,
)
from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from qtpy.QtGui import QKeyEvent

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import P, T


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

    def filter_key_events(self, widget: Any, receiver: ArrayView) -> Callable[[], None]:
        if not isinstance(widget, QWidget):
            raise TypeError(f"Expected widget to be QWidget, got {type(widget)}")

        f = KeyEventFilter(receiver)
        widget.installEventFilter(f)
        return lambda: widget.removeEventFilter(f)

    def process_events(self) -> None:
        """Process events for the application."""
        QApplication.processEvents()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        QTimer.singleShot(msec, Qt.TimerType.PreciseTimer, func)


_QT_KEY_MAP: dict[int, KeyCode] = {
    Qt.Key.Key_Up: KeyCode.UP,
    Qt.Key.Key_Down: KeyCode.DOWN,
    Qt.Key.Key_Left: KeyCode.LEFT,
    Qt.Key.Key_Right: KeyCode.RIGHT,
    Qt.Key.Key_Space: KeyCode.SPACE,
    Qt.Key.Key_Home: KeyCode.HOME,
    Qt.Key.Key_End: KeyCode.END,
}


def _qt_mods_to_keymods(modifiers: Qt.KeyboardModifier) -> KeyMod:
    mods = KeyMod.NONE
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        mods |= KeyMod.SHIFT
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        mods |= KeyMod.CTRL
    if modifiers & Qt.KeyboardModifier.AltModifier:
        mods |= KeyMod.ALT
    if modifiers & Qt.KeyboardModifier.MetaModifier:
        mods |= KeyMod.META
    return mods


class KeyEventFilter(QObject):
    def __init__(self, receiver: ArrayView) -> None:
        super().__init__()
        self.receiver = receiver

    def eventFilter(self, obj: QObject | None, qevent: QEvent | None) -> bool:
        if qevent is None or qevent.type() != QEvent.Type.KeyPress:
            return False

        key_event = cast("QKeyEvent", qevent)
        qt_key = key_event.key()
        key: KeyCode | str
        if qt_key in _QT_KEY_MAP:
            key = _QT_KEY_MAP[qt_key]
        else:
            text = key_event.text()
            if not text:
                return False
            key = text
        mods = _qt_mods_to_keymods(key_event.modifiers())
        self.receiver.keyPressed.emit(KeyPressEvent(key, mods))
        return False
