"""simple ndviewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"

from . import data
from .models import DataWrapper
from .util import imshow
from .viewers import ArrayViewer
from .views import run_app

__all__ = ["ArrayViewer", "DataWrapper", "NDViewer", "data", "imshow", "run_app"]
