"""Wrappers around GUI & graphics frameworks.

Most stuff in this module is not intended for public use, but [`ndv.views.bases`][]
shows the protocol that GUI & graphics classes should implement.
"""

from ._app import (
    CanvasBackend,
    GuiFrontend,
    call_later,
    get_array_canvas_class,
    get_array_view_class,
    get_histogram_canvas_class,
    gui_frontend,
    run_app,
)

__all__ = [
    "CanvasBackend",
    "GuiFrontend",
    "call_later",
    "get_array_canvas_class",
    "get_array_view_class",
    "get_histogram_canvas_class",
    "gui_frontend",
    "run_app",
]
