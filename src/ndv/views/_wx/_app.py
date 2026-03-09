from __future__ import annotations

from typing import TYPE_CHECKING, Any

import wx
from wx import (
    EVT_LEAVE_WINDOW,
    EVT_LEFT_DCLICK,
    EVT_LEFT_DOWN,
    EVT_LEFT_UP,
    EVT_MOTION,
    MouseEvent,
)

from ndv._types import (
    KeyCode,
    KeyMod,
    KeyPressEvent,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.views.bases._app import NDVApp

from ._main_thread import call_in_main_thread

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import P, T
    from ndv.views.bases._graphics._mouseable import Mouseable

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

    def filter_mouse_events(
        self, canvas: Any, receiver: Mouseable
    ) -> Callable[[], None]:
        if not isinstance(canvas, wx.Window):
            raise TypeError(f"Expected canvas to be wx.Window, got {type(canvas)}")

        # TIP: event.Skip() allows the event to propagate to other handlers.

        active_button: MouseButton = MouseButton.NONE

        def on_mouse_move(event: MouseEvent) -> None:
            nonlocal active_button
            nonlocal canvas

            mme = MouseMoveEvent(x=event.GetX(), y=event.GetY(), btn=active_button)
            if not receiver.on_mouse_move(mme):
                receiver.mouseMoved.emit(mme)
                event.Skip()
            # FIXME: get_cursor is VERY slow, unsure why.
            if cursor := receiver.get_cursor(mme):
                canvas.SetCursor(cursor.to_wx())

        def on_mouse_leave(event: MouseEvent) -> None:
            nonlocal active_button
            nonlocal canvas

            if not receiver.on_mouse_leave():
                event.Skip()
            receiver.mouseLeft.emit()

        def on_mouse_press(event: MouseEvent) -> None:
            nonlocal active_button

            # NB This function is bound to the left mouse button press
            active_button = MouseButton.LEFT
            mpe = MousePressEvent(x=event.GetX(), y=event.GetY(), btn=active_button)
            if not receiver.on_mouse_press(mpe):
                receiver.mousePressed.emit(mpe)
                event.Skip()

        def on_mouse_double_press(event: MouseEvent) -> None:
            nonlocal active_button

            # NB This function is bound to the left mouse button press
            active_button = MouseButton.LEFT
            mpe = MousePressEvent(x=event.GetX(), y=event.GetY(), btn=active_button)
            if not receiver.on_mouse_double_press(mpe):
                receiver.mouseDoublePressed.emit(mpe)
                event.Skip()

        def on_mouse_release(event: MouseEvent) -> None:
            nonlocal active_button

            mre = MouseReleaseEvent(x=event.GetX(), y=event.GetY(), btn=active_button)
            active_button = MouseButton.NONE
            if not receiver.on_mouse_release(mre):
                receiver.mouseReleased.emit(mre)
                event.Skip()

        canvas.Bind(EVT_MOTION, handler=on_mouse_move)
        canvas.Bind(EVT_LEAVE_WINDOW, handler=on_mouse_leave)
        canvas.Bind(EVT_LEFT_DOWN, handler=on_mouse_press)
        canvas.Bind(EVT_LEFT_DCLICK, handler=on_mouse_double_press)
        canvas.Bind(EVT_LEFT_UP, handler=on_mouse_release)

        def _unbind() -> None:
            canvas.Unbind(EVT_MOTION, handler=on_mouse_move)
            canvas.Unbind(EVT_LEAVE_WINDOW, handler=on_mouse_leave)
            canvas.Unbind(EVT_LEFT_DOWN, handler=on_mouse_press)
            canvas.Unbind(EVT_LEFT_DCLICK, handler=on_mouse_double_press)
            canvas.Unbind(EVT_LEFT_UP, handler=on_mouse_release)

        return _unbind

    def filter_key_events(
        self, widget: Any, canvas_widget: Any, receiver: ArrayView
    ) -> Callable[[], None]:
        if not isinstance(widget, wx.Window):
            raise TypeError(f"Expected widget to be wx.Window, got {type(widget)}")

        def on_key_down(event: wx.KeyEvent) -> None:
            key_code = event.GetKeyCode()
            key: KeyCode | str
            if key_code in _WX_KEY_MAP:
                key = _WX_KEY_MAP[key_code]
            else:
                uchar = event.GetUnicodeKey()
                if uchar != wx.WXK_NONE:
                    key = chr(uchar)
                else:
                    event.Skip()
                    return
            mods = KeyMod.NONE
            if event.ShiftDown():
                mods |= KeyMod.SHIFT
            if event.ControlDown():
                mods |= KeyMod.CTRL
            if event.AltDown():
                mods |= KeyMod.ALT
            if event.MetaDown():
                mods |= KeyMod.META
            receiver.keyPressed.emit(KeyPressEvent(key, mods))
            event.Skip()

        widget.Bind(wx.EVT_CHAR_HOOK, handler=on_key_down)

        def _unbind() -> None:
            widget.Unbind(wx.EVT_CHAR_HOOK, handler=on_key_down)

        return _unbind

    def process_events(self) -> None:
        """Process events."""
        wx.SafeYield()

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        wx.CallLater(msec, func)


_WX_KEY_MAP: dict[int, KeyCode] = {
    wx.WXK_UP: KeyCode.UP,
    wx.WXK_DOWN: KeyCode.DOWN,
    wx.WXK_LEFT: KeyCode.LEFT,
    wx.WXK_RIGHT: KeyCode.RIGHT,
    wx.WXK_SPACE: KeyCode.SPACE,
    wx.WXK_HOME: KeyCode.HOME,
    wx.WXK_END: KeyCode.END,
}
