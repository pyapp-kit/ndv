"""simple ndviewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@example.com"

from typing import TYPE_CHECKING

from . import data
from .util import imshow
from .viewer._data_wrapper import DataWrapper
from .viewer._viewer import NDViewer

__all__ = ["NDViewer", "DataWrapper", "imshow", "data"]


if TYPE_CHECKING:
    # these may be used externally, but are not guaranteed to be available at runtime
    # they must be used inside a TYPE_CHECKING block

    from .viewer._dims_slider import DimKey as DimKey
    from .viewer._dims_slider import Index as Index
    from .viewer._dims_slider import Indices as Indices
    from .viewer._dims_slider import Sizes as Sizes
