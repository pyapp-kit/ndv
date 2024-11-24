from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    from ndv.views.protocols import PCanvas

    from .protocols import PView

    GuiFrontend: TypeAlias = Literal["qt", "jupyter"]
    CanvasBackend: TypeAlias = Literal["vispy", "pygfx"]

GUI_ENV_VAR = "NDV_GUI_FRONTEND"
CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"


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


def _try_start_qapp() -> bool:
    try:
        from qtpy.QtWidgets import QApplication

        if QApplication.instance() is None:
            app = QApplication([])
            app.setOrganizationName("ndv")
            app.setApplicationName("ndv")

        return True
    except Exception:
        return False


def _determine_gui_frontend() -> GuiFrontend:
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    if requested:
        if requested not in ("qt", "jupyter"):
            raise ValueError(f"Invalid GUI frontend: {requested!r}")
        return cast("GuiFrontend", requested)
    if _is_running_in_notebook():
        return "jupyter"
    if _is_running_in_qapp():
        return "qt"
    if _try_start_qapp():
        return "qt"
    raise RuntimeError("Could not find an appropriate GUI frontend (Qt or Jupyter).")


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
