"""Wrappers around GUI & graphics frameworks.

Most stuff in this module is not intended for public use, but [`ndv.views.bases`][]
shows the protocol that GUI & graphics classes should implement.
"""

from ._app import (
    GuiFrontend,
    call_later,
    get_array_view_class,
    gui_frontend,
    process_events,
    run_app,
    set_canvas_backend,
    set_gui_backend,
)

__all__ = [
    "GuiFrontend",
    "call_later",
    "get_array_view_class",
    "gui_frontend",
    "process_events",
    "run_app",
    "set_canvas_backend",
    "set_gui_backend",
]
