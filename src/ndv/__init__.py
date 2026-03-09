"""Fast and flexible n-dimensional data viewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"

from . import data
from .controllers import ArrayViewer
from .models import DataWrapper
from .util import imshow
from .views import (
    call_later,
    process_events,
    run_app,
    set_canvas_backend,
    set_gui_backend,
)

__all__ = [
    "ArrayViewer",
    "DataWrapper",
    "call_later",
    "data",
    "imshow",
    "process_events",
    "run_app",
    "set_canvas_backend",
    "set_gui_backend",
]
