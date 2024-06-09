from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ndv.viewer._backends._protocols import PCanvas


def get_canvas(backend: str | None = None) -> type[PCanvas]:
    backend = backend or os.getenv("NDV_CANVAS_BACKEND", None)
    if backend == "vispy" or (backend is None and "vispy" in sys.modules):
        from ._vispy import VispyViewerCanvas

        return VispyViewerCanvas

    if backend == "pygfx" or (backend is None and "pygfx" in sys.modules):
        from ._pygfx import PyGFXViewerCanvas

        return PyGFXViewerCanvas

    if backend is None:
        if importlib.util.find_spec("vispy") is not None:
            from ._vispy import VispyViewerCanvas

            return VispyViewerCanvas

        if importlib.util.find_spec("pygfx") is not None:
            from ._pygfx import PyGFXViewerCanvas

            return PyGFXViewerCanvas

    raise RuntimeError("No canvas backend found")
