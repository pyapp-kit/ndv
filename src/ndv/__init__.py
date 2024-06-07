"""simple ndviewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@example.com"

from .viewer._indexing import DataWrapper
from .viewer._stack_viewer import NDViewer

__all__ = ["NDViewer", "DataWrapper"]
