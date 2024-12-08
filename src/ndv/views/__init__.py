from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    from ndv.views.protocols import PCanvas

    from .protocols import PView

    GuiFrontend: TypeAlias = Literal["qt", "jupyter", "wx"]
    CanvasBackend: TypeAlias = Literal["vispy", "pygfx"]

GUI_ENV_VAR = "NDV_GUI_FRONTEND"
CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
_APP_INSTANCE: Any | None = None


# TODO: add a way to set the frontend via an environment variable
# (for example, it should be possible to use qt frontend in a jupyter notebook)
def get_view_frontend_class() -> type[PView]:
    frontend = _determine_gui_frontend()
    if frontend == "jupyter":
        from ._jupyter.jupyter_view import JupyterViewerView

        return JupyterViewerView
    if frontend == "qt":
        from ._qt.qt_view import QtViewerView

        return QtViewerView

    if frontend == "wx":
        from ._wx.wx_view import WxViewerView

        return WxViewerView

    raise RuntimeError("No GUI frontend found")


def get_canvas_class(backend: str | None = None) -> type[PCanvas]:
    _backend = _determine_canvas_backend(backend)
    _frontend = _determine_gui_frontend()
    if _backend == "vispy":
        from vispy.app import use_app

        from ndv.views._vispy._vispy import VispyViewerCanvas

        if _frontend == "jupyter":
            use_app("jupyter_rfb")

        return VispyViewerCanvas

    if _backend == "pygfx":
        from ndv.views._pygfx._pygfx import PyGFXViewerCanvas

        return PyGFXViewerCanvas

    raise RuntimeError("No canvas backend found")


def _is_running_in_notebook() -> bool:
    if IPython := sys.modules.get("IPython"):
        if shell := IPython.get_ipython():
            return bool(shell.__class__.__name__ == "ZMQInteractiveShell")
    return False


def _is_running_in_qapp() -> bool:
    for mod_name in ("PyQt5", "PySide2", "PySide6", "PyQt6"):
        if mod := sys.modules.get(f"{mod_name}.QtWidgets"):
            if qapp := getattr(mod, "QApplication", None):
                return qapp.instance() is not None
    return False


def _is_running_in_wxapp() -> bool:
    if wx := sys.modules.get("wx"):
        return wx.App.Get() is not None
    return False


def _try_start_qapp() -> bool:
    global _APP_INSTANCE
    try:
        from qtpy.QtWidgets import QApplication

        if (qapp := QApplication.instance()) is None:
            qapp = QApplication([])
            qapp.setOrganizationName("ndv")
            qapp.setApplicationName("ndv")
        _APP_INSTANCE = qapp

        return True
    except Exception:
        return False


def _try_start_wxapp() -> bool:
    global _APP_INSTANCE
    try:
        import wx

        if (wxapp := wx.App.Get()) is None:
            wxapp = wx.App()
        _APP_INSTANCE = wxapp

        return True
    except Exception:
        return False


@cache  # not allowed to change
def _determine_gui_frontend() -> GuiFrontend:
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = ("qt", "jupyter", "wx")
    if requested:
        if requested not in valid:
            raise ValueError(f"Invalid GUI frontend: {requested!r}")
        return cast("GuiFrontend", requested)
    if _is_running_in_notebook():
        return "jupyter"
    if _is_running_in_qapp():
        return "qt"
    if _is_running_in_wxapp():
        return "wx"
    if _try_start_qapp():
        return "qt"
    if _try_start_wxapp():
        return "wx"
    raise RuntimeError(
        f"Could not find an appropriate GUI frontend: {valid}. "
        "Please pip install ndv[<frontend>] to pick one."
    )


def _determine_canvas_backend(requested: str | None) -> CanvasBackend:
    backend = requested or os.getenv(CANVAS_ENV_VAR, "").lower()

    if not backend:
        # first check for things that have already been imported
        if "vispy" in sys.modules:
            backend = "vispy"
        elif "pygfx" in sys.modules:
            backend = "pygfx"
        # then check for installed packages
        elif importlib.util.find_spec("vispy"):
            backend = "vispy"
        elif importlib.util.find_spec("pygfx"):
            backend = "pygfx"
        else:
            raise RuntimeError("No canvas backend found")

    if backend in ("vispy", "pygfx"):
        return cast("CanvasBackend", backend)

    raise ValueError(f"Invalid canvas backend: {backend!r}")


def run_app() -> None:
    frontend = _determine_gui_frontend()
    if frontend == "qt":
        from qtpy.QtWidgets import QApplication

        _try_start_qapp()
        if not isinstance(_APP_INSTANCE, QApplication):
            raise RuntimeError(
                f"Got unexpected application type: {type(_APP_INSTANCE)}"
            )
        _APP_INSTANCE.exec()
    elif frontend == "wx":
        import wx

        _try_start_wxapp()
        if not isinstance(_APP_INSTANCE, wx.App):
            raise RuntimeError(
                f"Got unexpected application type: {type(_APP_INSTANCE)}"
            )
        _APP_INSTANCE.MainLoop()
