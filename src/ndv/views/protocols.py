from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import cmap
from psygnal import Signal

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import numpy as np
    from psygnal import SignalInstance
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QWidget

    from ndv._types import AxisKey

from typing import Callable, Union, runtime_checkable


@runtime_checkable
class PSignalInstance(Protocol):
    """The protocol that a signal instance must implement.

    In practice this will either be a `pyqtSignal/pyqtBoundSignal` or a
    `psygnal.SignalInstance`.
    """

    def connect(self, slot: Callable) -> Any:
        """Connect slot to this signal."""

    def disconnect(self, slot: Callable | None = None) -> Any:
        """Disconnect slot from this signal.

        If `None`, all slots should be disconnected.
        """

    def emit(self, *args: Any) -> Any:
        """Emits the signal with the given arguments."""


@runtime_checkable
class PSignalDescriptor(Protocol):
    """Descriptor that returns a signal instance."""

    def __get__(self, instance: Any | None, owner: Any) -> PSignalInstance:
        """Returns the signal instance for this descriptor."""


PSignal = Union[PSignalDescriptor, PSignalInstance]


class PLutView(Protocol):
    visibleChanged: PSignal
    autoscaleChanged: PSignal
    cmapChanged: PSignal
    climsChanged: PSignal

    def set_name(self, name: str) -> None: ...
    def set_auto_scale(self, auto: bool) -> None: ...
    def set_colormap(self, cmap: cmap.Colormap) -> None: ...
    def set_clims(self, clims: tuple[float, float]) -> None: ...
    def set_lut_visible(self, visible: bool) -> None: ...


class LutView(Protocol):
    """An (interactive) view of a LookUp Table (LUT)."""

    cmapChanged: Signal = Signal(cmap.Colormap)
    gammaChanged: Signal = Signal(float)
    climsChanged: Signal = Signal(tuple[float, float])
    autoscaleChanged: Signal = Signal(object)

    def set_visibility(self, visible: bool) -> None:
        """Defines whether this view is visible.

        Properties
        ----------
        visible : bool
            True iff the view should be visible.
        """
        ...

    def set_cmap(self, lut: cmap.Colormap) -> None:
        """Defines the colormap backing the view.

        Properties
        ----------
        lut : cmap.Colormap
            The object mapping scalar values to RGB(A) colors.
        """
        ...

    def set_gamma(self, gamma: float) -> None:
        """Defines the exponent used for gamma correction.

        Properties
        ----------
        gamma : float
            The exponent used for gamma correction
        """
        ...

    def set_clims(self, clims: tuple[float, float]) -> None:
        """Defines the input clims.

        The contrast limits (clims) are the input values mapped to the minimum and
        maximum (respectively) of the LUT.

        Properties
        ----------
        clims : tuple[float, float]
            The clims
        """
        ...

    def set_autoscale(self, autoscale: bool | tuple[float, float]) -> None:
        """Defines whether autoscale has been enabled.

        Autoscale defines whether the contrast limits (clims) are adjusted when the
        data changes.

        Properties
        ----------
        autoscale : bool | tuple[float, float]
            If a boolean, true iff clims automatically changed on dataset alteration.
            If a tuple, indicated that clims automatically changed. Values denote
            the fraction of the dataset located below and above the lower and
            upper clims, respectively.
        """
        ...

    def view(self) -> Any:
        """The native object that can be displayed."""
        ...


class StatsView(Protocol):
    """A view of the statistics of a dataset."""

    def set_histogram(
        self, values: Sequence[float], bin_edges: Sequence[float]
    ) -> None:
        """Defines the distribution of the dataset.

        Properties
        ----------
        values : Sequence[int]
            A length (n) sequence of values representing clustered counts of data
            points. values[i] defines the number of data points falling between
            bin_edges[i] and bin_edges[i+1].
        bin_edges : Sequence[float]
            A length (n+1) sequence of values defining the intervals partitioning
            all data points. Must be non-decreasing.
        """
        ...

    def set_std_dev(self, std_dev: float) -> None:
        """Defines the standard deviation of the dataset.

        Properties
        ----------
        std_dev : float
            The standard deviation.
        """
        ...

    def set_average(self, avg: float) -> None:
        """Defines the average value of the dataset.

        Properties
        ----------
        std_dev : float
            The average value of the dataset.
        """
        ...

    def view(self) -> Any:
        """The native object that can be displayed."""
        ...


class HistogramView(StatsView, LutView):
    """A histogram-based view for LookUp Table (LUT) adjustment."""

    def set_domain(self, bounds: tuple[float, float] | None) -> None:
        """Sets the domain of the view.

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the x axis to the passed
            values. If None, sets them to the extent of the data instead.
        """
        ...

    def set_range(self, bounds: tuple[float, float] | None) -> None:
        """Sets the range of the view.

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the y axis to the passed
            values. If None, sets them to the extent of the data instead.
        """
        ...

    def set_vertical(self, vertical: bool) -> None:
        """Sets the axis of the domain.

        Properties
        ----------
        vertical : bool
            If true, views the domain along the y axis and the range along the x
            axis. If false, views the domain along the x axis and the range along
            the y axis.
        """
        ...

    def set_range_log(self, enabled: bool) -> None:
        """Sets the axis scale of the range.

        Properties
        ----------
        enabled : bool
            If true, the range will be displayed with a logarithmic (base 10)
            scale. If false, the range will be displayed with a linear scale.
        """
        ...


class CanvasElement(Protocol):
    """Protocol defining an interactive element on the Canvas."""

    @property
    def visible(self) -> bool:
        """Defines whether the element is visible on the canvas."""

    @visible.setter
    def visible(self, visible: bool) -> None:
        """Sets element visibility."""

    @property
    def can_select(self) -> bool:
        """Defines whether the element can be selected."""

    @property
    def selected(self) -> bool:
        """Returns element selection status."""

    @selected.setter
    def selected(self, selected: bool) -> None:
        """Sets element selection status."""

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        """Returns the element's cursor preference at the provided position."""

    def start_move(self, pos: Sequence[float]) -> None:
        """
        Behavior executed at the beginning of a "move" operation.

        In layman's terms, this is the behavior executed during the the "click"
        of a "click-and-drag".
        """

    def move(self, pos: Sequence[float]) -> None:
        """
        Behavior executed throughout a "move" operation.

        In layman's terms, this is the behavior executed during the "drag"
        of a "click-and-drag".
        """

    def remove(self) -> None:
        """Removes the element from the canvas."""


class PImageHandle(CanvasElement, Protocol):
    @property
    def data(self) -> np.ndarray: ...
    @data.setter
    def data(self, data: np.ndarray) -> None: ...
    @property
    def clim(self) -> Any: ...
    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None: ...
    @property
    def cmap(self) -> cmap.Colormap: ...
    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None: ...


class PView(Protocol):
    """Protocol that front-end viewers must implement."""

    currentIndexChanged: PSignal
    resetZoomClicked: PSignal
    mouseMoved: PSignal  # Signal(_types.MouseMoveEvent)

    def __init__(self, canvas_widget: Any, **kwargs: Any) -> None: ...
    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int | slice]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def set_data_info(self, data_info: str) -> None:
        """Set info about the currently displayed data, usually above canvas."""

    def set_hover_info(self, hover_info: str) -> None:
        """Set info about the current hover position, usually below canvas."""

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...
    def add_lut_view(self) -> PLutView: ...
    def show(self) -> None: ...

    # def refresh(self) -> None: ...
    # def add_image_to_canvas(self, data: Any) -> PImageHandle: ...


class PRoiHandle(CanvasElement, Protocol):
    @property
    def vertices(self) -> Sequence[Sequence[float]]: ...
    @vertices.setter
    def vertices(self, data: Sequence[Sequence[float]]) -> None: ...
    @property
    def color(self) -> Any: ...
    @color.setter
    def color(self, color: cmap.Color) -> None: ...
    @property
    def border_color(self) -> Any: ...
    @border_color.setter
    def border_color(self, color: cmap.Color) -> None: ...


class PCanvas(Protocol):
    def __init__(self) -> None: ...
    def set_ndim(self, ndim: Literal[2, 3]) -> None: ...
    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = ...,
    ) -> None: ...
    def refresh(self) -> None: ...
    def qwidget(self) -> QWidget: ...
    def add_image(
        self, data: np.ndarray | None = ..., cmap: cmap.Colormap | None = ...
    ) -> PImageHandle: ...
    def add_volume(
        self, data: np.ndarray | None = ..., cmap: cmap.Colormap | None = ...
    ) -> PImageHandle: ...
    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""

    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]: ...
    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> PRoiHandle: ...
