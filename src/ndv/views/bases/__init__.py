"""Abstract base classes for views and viewable objects."""

from typing import Any

from ._app import NDVApp
from ._array_view import ArrayView
from ._lut_view import LUTView
from ._view_base import Viewable

__all__ = [
    "ArrayView",
    "LUTView",
    "NDVApp",
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
