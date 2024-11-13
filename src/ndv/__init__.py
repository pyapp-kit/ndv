"""simple ndviewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@example.com"

from . import data
from ._old_viewer import NDViewer
from .models import DataWrapper
from .util import imshow

__all__ = ["NDViewer", "DataWrapper", "imshow", "data"]
