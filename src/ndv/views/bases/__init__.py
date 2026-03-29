"""Abstract base classes for views and viewable objects."""

from typing import Any

from ._app import NDVApp
from ._array_view import ArrayView
from ._graphics._canvas import ArrayCanvas, HistogramCanvas, SharedHistogramCanvas
from ._graphics._canvas_elements import CanvasElement, ImageHandle, RectangularROIHandle
from ._graphics._mouseable import Mouseable
from ._lut_view import LUTView
from ._view_base import Viewable

__all__ = [
    "ArrayCanvas",
    "ArrayView",
    "CanvasElement",
    "HistogramCanvas",
    "ImageHandle",
    "LUTView",
    "Mouseable",
    "NDVApp",
    "RectangularROIHandle",
    "SharedHistogramCanvas",
    "Viewable",
]


def __getattr__(name: str) -> Any:
    if name == "LutView":
        import warnings

        warnings.warn(
            "LutView is deprecated, use LUTView instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LUTView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
