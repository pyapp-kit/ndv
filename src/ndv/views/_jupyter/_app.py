from __future__ import annotations

import os
from types import MethodType
from typing import TYPE_CHECKING, Any

from jupyter_rfb import RemoteFrameBuffer

from ndv._types import (
    KeyCode,
    KeyMod,
    KeyPressEvent,
)
from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
    from collections.abc import Callable

    from ndv.views.bases import ArrayView


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

    def filter_key_events(self, widget: Any, receiver: ArrayView) -> Callable[[], None]:
        # In Jupyter, key events must go through the RemoteFrameBuffer canvas
        # (which uses a hidden <input> element to capture keys), not the
        # ipywidgets container. Walk the widget tree to find it.
        target = _find_rfb(widget)
        if target is None:
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
    """Walk the ipywidgets tree to find a RemoteFrameBuffer child."""
    if isinstance(widget, RemoteFrameBuffer):
        return widget
    for child in getattr(widget, "children", ()):
        if found := _find_rfb(child):
            return found
    return None
