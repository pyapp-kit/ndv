"""Fast and flexible n-dimensional data viewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"

from . import data
from .controllers import ArrayViewer, StreamingViewer
from .models import DataWrapper
from .util import imshow
from .views import call_later, run_app

__all__ = [
    "ArrayViewer",
    "DataWrapper",
    "StreamingViewer",
    "call_later",
    "data",
    "imshow",
    "run_app",
]
