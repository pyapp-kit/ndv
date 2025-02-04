from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import wx
from wx import EVT_LEFT_DOWN, EVT_LEFT_UP, EVT_MOTION, EvtHandler, MouseEvent

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.views.bases._app import NDVApp

from ._main_thread import call_in_main_thread

if TYPE_CHECKING:
    from concurrent.futures import Future

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import P, T
    from ndv.views.bases._graphics._mouseable import Mouseable


class WxAppWrap(NDVApp):
    """Provider for wxPython."""

    IPY_MAGIC_KEY = "wx"

    def create_app(self) -> Any:
        if (wxapp := wx.App.Get()) is None:
            wxapp = wx.App()

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

    def filter_mouse_events(
        self, canvas: Any, receiver: Mouseable
    ) -> Callable[[], None]:
        if not isinstance(canvas, EvtHandler):
            raise TypeError(
                f"Expected vispy canvas to be wx EvtHandler, got {type(canvas)}"
            )

        # TIP: event.Skip() allows the event to propagate to other handlers.

        def on_mouse_move(event: MouseEvent) -> None:
            mme = MouseMoveEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_move(mme):
                receiver.mouseMoved.emit(mme)
                event.Skip()

        def on_mouse_press(event: MouseEvent) -> None:
            mpe = MousePressEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_press(mpe):
                receiver.mousePressed.emit(mpe)
                event.Skip()

        def on_mouse_release(event: MouseEvent) -> None:
            mre = MouseReleaseEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_release(mre):
                receiver.mouseReleased.emit(mre)
                event.Skip()

        canvas.Bind(EVT_MOTION, on_mouse_move)
        canvas.Bind(EVT_LEFT_DOWN, on_mouse_press)
        canvas.Bind(EVT_LEFT_UP, on_mouse_release)

        def _unbind() -> None:
            canvas.Unbind(EVT_MOTION, on_mouse_move)
            canvas.Unbind(EVT_LEFT_DOWN, on_mouse_press)
            canvas.Unbind(EVT_LEFT_UP, on_mouse_release)

        return _unbind
