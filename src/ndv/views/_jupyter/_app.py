from __future__ import annotations

import os
from types import MethodType
from typing import TYPE_CHECKING, Any

from jupyter_rfb import RemoteFrameBuffer

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ndv.views.bases import ArrayView
    from ndv.views.bases._graphics._canvas import GraphicsCanvas
    from ndv.views.bases._graphics._mouseable import Mouseable


class JupyterAppWrap(NDVApp):
    """Provider for Jupyter notebooks/lab (NOT ipython)."""

    def is_running(self) -> bool:
        if ipy_shell := self._ipython_shell():
            return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
        return False

    def create_app(self) -> Any:
        if not self.is_running() and not os.getenv("PYTEST_CURRENT_TEST"):
            # if we got here, it probably means that someone used
            # NDV_GUI_FRONTEND=jupyter without actually being in a jupyter notebook
            # we allow it in tests, but not in normal usage.
            raise RuntimeError(  # pragma: no cover
                "Jupyter is not running a notebook shell.  Cannot create app."
            )

        # No app creation needed...
        # but make sure we can actually import the stuff we need
        import ipywidgets  # noqa: F401
        import jupyter  # noqa: F401

    def array_view_class(self) -> type[ArrayView]:
        from ._array_view import JupyterArrayView

        return JupyterArrayView

    @staticmethod
    def mouse_btn(btn: Any) -> MouseButton:
        if btn == 1:
            return MouseButton.LEFT
        if btn == 2:
            return MouseButton.RIGHT
        if btn == 3:
            return MouseButton.MIDDLE
        return MouseButton.NONE

    def filter_mouse_events(
        self, canvas: Any, receiver: Mouseable
    ) -> Callable[[], None]:
        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )

        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = canvas.handle_event
        active_btn: MouseButton = MouseButton.NONE

        def handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            nonlocal active_btn

            intercepted = False
            etype = ev["event_type"]
            if etype == "pointer_move":
                mme = MouseMoveEvent(x=ev["x"], y=ev["y"], btn=active_btn)
                intercepted |= receiver.on_mouse_move(mme)
                if cursor := receiver.get_cursor(mme):
                    canvas.cursor = cursor.to_jupyter()
                receiver.mouseMoved.emit(mme)
            elif etype == "pointer_down":
                if "button" in ev:
                    active_btn = JupyterAppWrap.mouse_btn(ev["button"])
                else:
                    active_btn = MouseButton.NONE
                mpe = MousePressEvent(x=ev["x"], y=ev["y"], btn=active_btn)
                intercepted |= receiver.on_mouse_press(mpe)
                receiver.mousePressed.emit(mpe)
            elif etype == "double_click":
                # Note that in Jupyter, a double_click event is not a pointer event
                # and as such, we need to handle both press and release. See
                # https://github.com/vispy/jupyter_rfb/blob/62831dd5a87bc19b4fd5f921d802ed21141e61ec/js/lib/widget.js#L270
                btn = JupyterAppWrap.mouse_btn(ev["button"])
                mpe = MousePressEvent(x=ev["x"], y=ev["y"], btn=btn)
                intercepted |= receiver.on_mouse_double_press(mpe)
                receiver.mouseDoublePressed.emit(mpe)
                # Release
                mre = MouseReleaseEvent(x=ev["x"], y=ev["y"], btn=btn)
                intercepted |= receiver.on_mouse_release(mre)
                receiver.mouseReleased.emit(mre)
            elif etype == "pointer_up":
                mre = MouseReleaseEvent(x=ev["x"], y=ev["y"], btn=active_btn)
                active_btn = MouseButton.NONE
                intercepted |= receiver.on_mouse_release(mre)
                receiver.mouseReleased.emit(mre)

            if not intercepted:
                super_handle_event(ev)

        canvas.handle_event = MethodType(handle_event, canvas)
        return lambda: setattr(canvas, "handle_event", super_handle_event)

    def filter_key_events(
        self, canvas: Any, receiver: GraphicsCanvas
    ) -> Callable[[], None]:
        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )

        super_handle_event = canvas.handle_event

        def handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            if ev["event_type"] == "key_down":
                key_str = ev.get("key", "")
                key: KeyCode | str
                if key_str in _JUPYTER_KEY_MAP:
                    key = _JUPYTER_KEY_MAP[key_str]
                elif len(key_str) == 1:
                    key = key_str
                else:
                    super_handle_event(ev)
                    return
                mods = KeyMod.NONE
                if ev.get("shiftKey"):
                    mods |= KeyMod.SHIFT
                if ev.get("ctrlKey"):
                    mods |= KeyMod.CTRL
                if ev.get("altKey"):
                    mods |= KeyMod.ALT
                if ev.get("metaKey"):
                    mods |= KeyMod.META
                receiver.keyPressed.emit(KeyPressEvent(key, mods))
            super_handle_event(ev)

        canvas.handle_event = MethodType(handle_event, canvas)

        original_super = super_handle_event

        def _restore() -> None:
            # restore to the handler that was present before we patched
            canvas.handle_event = original_super

        return _restore


_JUPYTER_KEY_MAP: dict[str, KeyCode] = {
    "ArrowUp": KeyCode.UP,
    "ArrowDown": KeyCode.DOWN,
    "ArrowLeft": KeyCode.LEFT,
    "ArrowRight": KeyCode.RIGHT,
    " ": KeyCode.SPACE,
    "Home": KeyCode.HOME,
    "End": KeyCode.END,
}
