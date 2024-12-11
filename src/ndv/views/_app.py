from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from ndv.views.bases import ArrayCanvas, ArrayView, HistogramCanvas


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
def get_view_frontend_class() -> type[ArrayView]:
    frontend = gui_frontend()
    if frontend == GuiFrontend.QT:
        from ._qt.qt_view import QtArrayView

        return QtArrayView

    if frontend == GuiFrontend.JUPYTER:
        from ._jupyter.jupyter_view import JupyterArrayView

        return JupyterArrayView

    raise RuntimeError("No GUI frontend found")


def get_canvas_class(backend: str | None = None) -> type[ArrayCanvas]:
    _backend = _determine_canvas_backend(backend)
    if _backend == CanvasBackend.VISPY:
        from vispy.app import use_app

        from ndv.views._vispy._vispy import VispyViewerCanvas

        if gui_frontend() == GuiFrontend.JUPYTER:
            use_app("jupyter_rfb")

        return VispyViewerCanvas

    if _backend == CanvasBackend.PYGFX:
        from ndv.views._pygfx._pygfx import GfxArrayCanvas

        return GfxArrayCanvas

    raise RuntimeError(f"No canvas backend found for {_backend}")


def get_histogram_canvas_class(backend: str | None = None) -> type[HistogramCanvas]:
    _backend = _determine_canvas_backend(backend)
    if _backend == CanvasBackend.VISPY:
        from ndv.views._vispy._histogram import VispyHistogramCanvas

        return VispyHistogramCanvas
    raise RuntimeError(f"Histogram not supported for backend: {_backend}")


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

        _install_excepthook()
        _APP_INSTANCE = qapp
        return True
    except Exception:
        return False


def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit().

    This is necessary to prevent the application from closing when an exception
    is raised.
    """
    sys.excepthook = _no_exit_excepthook


def _no_exit_excepthook(
    type: type[BaseException], value: BaseException, tb: TracebackType | None
) -> None:
    """Excepthook that prints the traceback to the console.

    By default, Qt's excepthook raises sys.exit(), which is not what we want.
    """
    # this could be elaborated to do all kinds of things...
    print("\n-----------------------")
    traceback.print_exception(type, value, tb)


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
    if frontend == GuiFrontend.QT:
        from qtpy.QtWidgets import QApplication

        _try_start_qapp()
        if not isinstance(_APP_INSTANCE, QApplication):
            raise RuntimeError(
                f"Got unexpected application type: {type(_APP_INSTANCE)}"
            )
        _APP_INSTANCE.exec()
    elif frontend == GuiFrontend.JUPYTER:
        pass  # nothing to do here
