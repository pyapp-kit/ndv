from __future__ import annotations

from typing import TYPE_CHECKING, Any

import wx

from ndv.views.bases._app import NDVApp

from ._main_thread import call_in_main_thread

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import P, T

_app = None


class WxAppWrap(NDVApp):
    """Provider for wxPython."""

    IPY_MAGIC_KEY = "wx"

    def create_app(self) -> Any:
        global _app
        if (wxapp := wx.App.Get()) is None:
            _app = wxapp = wx.App()

        self._maybe_enable_ipython_gui()
        self._install_excepthook()
        return wxapp

    def run(self) -> None:
        app = wx.App.Get() or self.create_app()

        if ipy_shell := self._ipython_shell():
            # if we're already in an IPython session with %gui qt, don't block
            if str(ipy_shell.active_eventloop).startswith("wx"):
                return

        app.MainLoop()

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        return call_in_main_thread(func, *args, **kwargs)

    def array_view_class(self) -> type[ArrayView]:
        from ._array_view import WxArrayView

        return WxArrayView

    def process_events(self) -> None:
        """Process events."""
        wx.SafeYield()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        wx.CallLater(msec, func)
