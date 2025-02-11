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
from .views import run_app

__all__ = ["ArrayViewer", "DataWrapper", "StreamingViewer", "data", "imshow", "run_app"]
