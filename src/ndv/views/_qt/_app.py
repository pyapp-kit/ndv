from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication

from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

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

    def process_events(self) -> None:
        """Process events for the application."""
        QApplication.processEvents()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        QTimer.singleShot(msec, Qt.TimerType.PreciseTimer, func)
