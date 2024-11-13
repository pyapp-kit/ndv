from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ndv.views.protocols import PCanvas

    from .protocols import PView


def get_view_backend() -> PView:
    if _is_running_in_notebook():
        from ._jupyter.jupyter_view import JupyterViewerView

        return JupyterViewerView()
    elif _is_running_in_qapp():
        from ._qt.qt_view import QViewerView

        return QViewerView()

    raise RuntimeError("Could not determine the appropriate viewer backend")


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


def get_canvas_class(backend: str | None = None) -> type[PCanvas]:
    backend = backend or os.getenv("NDV_CANVAS_BACKEND", None)
    if backend == "vispy" or (backend is None and "vispy" in sys.modules):
        from ndv.views._vispy._vispy import VispyViewerCanvas

        return VispyViewerCanvas

    if backend == "pygfx" or (backend is None and "pygfx" in sys.modules):
        from ndv.views._pygfx._pygfx import PyGFXViewerCanvas

        return PyGFXViewerCanvas

    if backend is None:
        if importlib.util.find_spec("vispy") is not None:
            from ndv.views._vispy._vispy import VispyViewerCanvas

            return VispyViewerCanvas

        if importlib.util.find_spec("pygfx") is not None:
            from ndv.views._pygfx._pygfx import PyGFXViewerCanvas

            return PyGFXViewerCanvas

    raise RuntimeError("No canvas backend found")
