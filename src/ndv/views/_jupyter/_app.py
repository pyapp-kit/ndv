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
    from ndv.views.bases._graphics._mouseable import Mouseable


class JupyterAppWrap(NDVApp):
    """Provider for Jupyter notebooks/lab (NOT ipython)."""

    def is_running(self) -> bool:
        if ipy_shell := self._ipython_shell():
            return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
        return False

    def create_app(self) -> Any:
        if not self.is_running() and not os.getenv("PYTEST_CURRENT_TEST"):
            raise RuntimeError(  # pragma: no cover
                "Jupyter is not running a notebook shell.  Cannot create app."
            )

        import anywidget  # noqa: F401
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
                btn = JupyterAppWrap.mouse_btn(ev["button"])
                mpe = MousePressEvent(x=ev["x"], y=ev["y"], btn=btn)
                intercepted |= receiver.on_mouse_double_press(mpe)
                receiver.mouseDoublePressed.emit(mpe)
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

    def filter_key_events(self, widget: Any, receiver: ArrayView) -> Callable[[], None]:
        # The widget is NdvWidgetState. Get the canvas from its _canvas_ref.
        target = getattr(widget, "_canvas_ref", None)
        if target is None:
            target = _find_rfb(widget)
        if target is None or not isinstance(target, RemoteFrameBuffer):
            return lambda: None

        super_handle_event = target.handle_event

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

        target.handle_event = MethodType(handle_event, target)
        return lambda: setattr(target, "handle_event", super_handle_event)


_JUPYTER_KEY_MAP: dict[str, KeyCode] = {
    "ArrowUp": KeyCode.UP,
    "ArrowDown": KeyCode.DOWN,
    "ArrowLeft": KeyCode.LEFT,
    "ArrowRight": KeyCode.RIGHT,
    " ": KeyCode.SPACE,
    "Home": KeyCode.HOME,
    "End": KeyCode.END,
}


def _find_rfb(widget: Any) -> RemoteFrameBuffer | None:
    """Walk widget attributes to find a RemoteFrameBuffer."""
    if isinstance(widget, RemoteFrameBuffer):
        return widget
    # Check anywidget state's canvas ref
    canvas = getattr(widget, "_canvas_ref", None)
    if isinstance(canvas, RemoteFrameBuffer):
        return canvas
    # Fallback: walk ipywidgets children if present
    for child in getattr(widget, "children", ()):
        if found := _find_rfb(child):
            return found
    return None
