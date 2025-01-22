from __future__ import annotations

import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from jupyter_rfb import RemoteFrameBuffer

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
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
            # if we got here, it probably means that someone used
            # NDV_GUI_FRONTEND=jupyter without actually being in a juptyer notebook
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

        def handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            etype = ev["event_type"]
            if etype == "pointer_move":
                mme = MouseMoveEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_move(mme)
                receiver.mouseMoved.emit(mme)
            elif etype == "pointer_down":
                mpe = MousePressEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_press(mpe)
                receiver.mousePressed.emit(mpe)
            elif etype == "pointer_up":
                mre = MouseReleaseEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_release(mre)
                receiver.mouseReleased.emit(mre)
            super_handle_event(ev)

        canvas.handle_event = MethodType(handle_event, canvas)
        return lambda: setattr(canvas, "handle_event", super_handle_event)
