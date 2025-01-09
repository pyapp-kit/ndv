"""simple ndviewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"

from . import data
from ._views import run_app
from .controllers import ArrayViewer
from .models import DataWrapper
from .util import imshow

__all__ = ["ArrayViewer", "DataWrapper", "data", "imshow", "run_app"]
