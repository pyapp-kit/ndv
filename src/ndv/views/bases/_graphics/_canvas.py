from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
from psygnal import Signal

from ndv.views.bases._lut_view import LUTView
from ndv.views.bases._view_base import Viewable

from ._mouseable import Mouseable

if TYPE_CHECKING:
    import numpy as np

    from ndv._types import ChannelKey
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
    ) -> None:
        """Sets the bounds of the camera."""
        ...

    def zoom(self, factor: float | tuple, center: tuple[float, float]) -> None:
        """Zoom in (or out) at the given center (world coordinates)."""

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

    def set_scales(self, scales: tuple[float, ...]) -> None:
        """Set per-visible-axis scale factors for rendering."""


class HistogramCanvas(GraphicsCanvas, LUTView):
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

    def highlight(self, value: float | None) -> None:
        """Highlights a domain value on the histogram."""


class SharedHistogramCanvas(GraphicsCanvas):
    """Multi-channel overlay histogram with per-channel clim/gamma controls."""

    climsChanged = Signal(object, tuple)
    gammaChanged = Signal(object, float)

    def set_channel_data(
        self, key: ChannelKey, counts: np.ndarray, bin_edges: np.ndarray
    ) -> None:
        """Set or update histogram data for a channel."""

    def set_channel_color(self, key: ChannelKey, color: tuple) -> None:
        """Set the display color (RGBA) for a channel."""

    def set_channel_visible(self, key: ChannelKey, visible: bool) -> None:
        """Show or hide a channel on the histogram."""

    def set_channel_clims(self, key: ChannelKey, clims: tuple[float, float]) -> None:
        """Update the clim line positions for a channel."""

    def set_channel_gamma(self, key: ChannelKey, gamma: float) -> None:
        """Update the gamma curve for a channel."""

    def remove_channel(self, key: ChannelKey) -> None:
        """Remove a channel from the histogram."""

    def set_channel_name(self, key: ChannelKey, name: str) -> None:
        """Set the display name for a channel (used in legend)."""

    def set_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        """Set global bounds for clim values and x-axis range."""

    def set_log_base(self, base: float | None) -> None:
        """Set logarithmic scale base, or None for linear."""

    def highlight(self, channel_values: dict[object, float]) -> None:
        """Highlight domain values across channels."""
