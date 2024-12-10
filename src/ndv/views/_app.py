from __future__ import annotations

import importlib.util
import os
import sys
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ndv.views.protocols import PCanvas

    from .protocols import PView


GUI_ENV_VAR = "NDV_GUI_FRONTEND"
CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
_APP_INSTANCE: Any | None = None  # pointer to the global app instsance.


class GuiFrontend(str, Enum):
    QT = "qt"
    JUPYTER = "jupyter"


class CanvasBackend(str, Enum):
    VISPY = "vispy"
    PYGFX = "pygfx"


# TODO: add a way to set the frontend via an environment variable
# (for example, it should be possible to use qt frontend in a jupyter notebook)
def get_view_frontend_class() -> type[PView]:
    frontend = gui_frontend()
    if frontend == "jupyter":
        from ._jupyter.jupyter_view import JupyterViewerView

        return JupyterViewerView
    if frontend == "qt":
        from ._qt.qt_view import QtViewerView

        return QtViewerView

    raise RuntimeError("No GUI frontend found")


def get_canvas_class(backend: str | None = None) -> type[PCanvas]:
    _backend = _determine_canvas_backend(backend)
    _frontend = gui_frontend()
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


@cache  # not allowed to change
def gui_frontend() -> GuiFrontend:
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = {x.value for x in GuiFrontend}
    if requested:
        if requested not in valid:
            raise ValueError(
                f"Invalid GUI frontend: {requested!r}. Valid options: {valid}"
            )
        return GuiFrontend(requested)
    if _is_running_in_notebook():
        return GuiFrontend.JUPYTER
    if _is_running_in_qapp():
        return GuiFrontend.QT
    if _try_start_qapp():
        return GuiFrontend.QT
    raise RuntimeError(f"Could not find an appropriate GUI frontend: {valid!r}")


def _determine_canvas_backend(requested: str | None) -> CanvasBackend:
    backend = requested or os.getenv(CANVAS_ENV_VAR, "").lower()
    valid = {x.value for x in CanvasBackend}
    if backend:
        if backend not in valid:
            raise ValueError(
                f"Invalid canvas backend: {backend!r}. Valid options: {valid}"
            )
        return CanvasBackend(backend)

    # first check for things that have already been imported
    if "vispy" in sys.modules:
        return CanvasBackend.VISPY
    if "pygfx" in sys.modules:
        return CanvasBackend.PYGFX
    # then check for installed packages
    if importlib.util.find_spec("vispy"):
        return CanvasBackend.VISPY
    if importlib.util.find_spec("pygfx"):
        return CanvasBackend.PYGFX
    raise RuntimeError(f"Could not find an appropriate canvas backend: {valid!r}")


def run_app() -> None:
    """Start the GUI application event loop."""
    frontend = gui_frontend()
    if frontend == "qt":
        from qtpy.QtWidgets import QApplication

        _try_start_qapp()
        if not isinstance(_APP_INSTANCE, QApplication):
            raise RuntimeError(
                f"Got unexpected application type: {type(_APP_INSTANCE)}"
            )
        _APP_INSTANCE.exec()
    elif frontend == "jupyter":
        pass  # nothing to do here
