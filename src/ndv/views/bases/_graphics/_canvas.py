from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np

from ndv.views.bases._lut_view import LutView
from ndv.views.bases._view_base import Viewable

from ._mouseable import Mouseable

if TYPE_CHECKING:
    import numpy as np

    from ndv.models._viewer_model import ArrayViewerModel

    from ._canvas_elements import CanvasElement, ImageHandle, RectangularROIHandle


class GraphicsCanvas(Viewable, Mouseable):
    """ABC for graphics canvas providers."""

    @abstractmethod
    def refresh(self) -> None: ...
    @abstractmethod
    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = ...,
    ) -> None: ...
    @abstractmethod
    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""

    @abstractmethod
    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]: ...


# TODO: These classes will probably be merged and refactored in the future.


class ArrayCanvas(GraphicsCanvas):
    """ABC for canvases that show array data."""

    @abstractmethod
    def __init__(self, viewer_model: ArrayViewerModel | None = ...) -> None: ...
    @abstractmethod
    def set_ndim(self, ndim: Literal[2, 3]) -> None: ...
    @abstractmethod
    @abstractmethod
    def add_image(self, data: np.ndarray | None = ...) -> ImageHandle: ...
    @abstractmethod
    def add_volume(self, data: np.ndarray | None = ...) -> ImageHandle: ...
    @abstractmethod
    def add_bounding_box(self) -> RectangularROIHandle: ...


class HistogramCanvas(GraphicsCanvas, LutView):
    """A histogram-based view for LookUp Table (LUT) adjustment."""

    def set_vertical(self, vertical: bool) -> None:
        """If True, orient axes vertically (x-axis on left)."""

    def set_log_base(self, base: float | None) -> None:
        """Sets the axis scale of the range.

        Properties
        ----------
        enabled : bool
            If true, the range will be displayed with a logarithmic (base 10)
            scale. If false, the range will be displayed with a linear scale.
        """

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Sets the histogram data.

        Properties
        ----------
        values : np.ndarray
            The histogram values.
        bin_edges : np.ndarray
            The bin edges of the histogram.
        """
