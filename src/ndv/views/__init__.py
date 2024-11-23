from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from ndv.views.protocols import PCanvas

    from .protocols import PView


def get_view_frontend_class() -> type[PView]:
    if _is_running_in_notebook():
        from ._jupyter.jupyter_view import JupyterViewerView

        return JupyterViewerView
    elif _is_running_in_qapp():
        from ._qt.qt_view import QViewerView

        return QViewerView

    raise RuntimeError("Could not determine an appropriate view frontend")


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


def _determine_canvas_backend(requested: str | None) -> Literal["vispy", "pygfx"]:
    backend = requested or os.getenv("NDV_CANVAS_BACKEND", "").lower()

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

    if backend not in ("vispy", "pygfx"):
        raise ValueError(f"Invalid canvas backend: {backend!r}")
    return cast(Literal["vispy", "pygfx"], backend)


def get_canvas_class(backend: str | None = None) -> type[PCanvas]:
    _backend = _determine_canvas_backend(backend)
    if _backend == "vispy":
        from ndv.views._vispy._vispy import VispyViewerCanvas

        return VispyViewerCanvas

    if _backend == "pygfx":
        from ndv.views._pygfx._pygfx import PyGFXViewerCanvas

        return PyGFXViewerCanvas

    raise RuntimeError("No canvas backend found")
