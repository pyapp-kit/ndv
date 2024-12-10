from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Union

from psygnal import Signal

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap
    import numpy as np
    from qtpy.QtCore import Qt

    from ndv._types import AxisKey
    from ndv.models._array_display_model import ChannelMode


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


class PSignalDescriptor(Protocol):
    """Descriptor that returns a signal instance."""

    def __get__(self, instance: Any | None, owner: Any) -> PSignalInstance:
        """Returns the signal instance for this descriptor."""


PSignal = Union[PSignalDescriptor, PSignalInstance]


class PLutView(Protocol):
    """An (interactive) view of a LookUp Table (LUT)."""

    visibleChanged: PSignal
    autoscaleChanged: PSignal
    cmapChanged: PSignal
    climsChanged: PSignal
    gammaChanged: PSignal

    def set_name(self, name: str) -> None:
        """Defines the name of the view.

        Properties
        ----------
        name : str
            The name (label) of the LUT
        """

    def set_auto_scale(self, auto: bool) -> None:
        """Defines whether autoscale has been enabled.

        Autoscale defines whether the contrast limits (clims) are adjusted when the
        data changes.

        Properties
        ----------
        autoscale : bool
            True iff clims automatically changed on dataset alteration.
        """

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        """Defines the colormap backing the view.

        Properties
        ----------
        lut : cmap.Colormap
            The object mapping scalar values to RGB(A) colors.
        """

    def set_clims(self, clims: tuple[float, float]) -> None:
        """Defines the input clims.

        The contrast limits (clims) are the input values mapped to the minimum and
        maximum (respectively) of the LUT.

        Properties
        ----------
        clims : tuple[float, float]
            The clims
        """

    def set_lut_visible(self, visible: bool) -> None:
        """Defines whether this view is visible.

        Properties
        ----------
        visible : bool
            True iff the view should be visible.
        """

    def set_gamma(self, gamma: float) -> None:
        """Defines the input gamma.

        properties
        ----------
        gamma : float
            The gamma
        """

    def setVisible(self, visible: bool) -> None:
        """Sets the visibility of the view/widget itself."""


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

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
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
    """Primary protocol for top level, front-end viewers."""

    currentIndexChanged: PSignal
    resetZoomClicked: PSignal
    channelModeChanged: PSignal

    def __init__(
        self,
        canvas_widget: Any,
        histogram_widget: Any,
        **kwargs: Any,
    ) -> None: ...
    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int | slice]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def set_channel_mode(self, mode: ChannelMode) -> None: ...
    def set_data_info(self, data_info: str) -> None:
        """Set info about the currently displayed data, usually above canvas."""

    def set_hover_info(self, hover_info: str) -> None:
        """Set info about the current hover position, usually below canvas."""

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...
    def add_lut_view(self) -> PLutView: ...
    def remove_lut_view(self, view: PLutView) -> None: ...
    def show(self) -> None: ...
    def hide(self) -> None: ...


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


# TODO: it's becoming confusing whether these are meant to be protocols or bases


class Mouseable(Protocol):
    mouseMoved: PSignal = Signal(MouseMoveEvent)
    mousePressed: PSignal = Signal(MousePressEvent)
    mouseReleased: PSignal = Signal(MouseReleaseEvent)

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False


class PCanvas(Mouseable, Protocol):
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
    def frontend_widget(self) -> Any: ...
    def add_image(
        self,
        data: np.ndarray | None = ...,
        cmap: cmap.Colormap | None = ...,
        clims: tuple[float, float] | None = ...,
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


class PHistogramCanvas(PLutView, Mouseable, Protocol):
    """A histogram-based view for LookUp Table (LUT) adjustment."""

    # TODO: Remove?
    def refresh(self) -> None: ...

    def frontend_widget(self) -> Any:
        """Returns an object understood by the widget frontend."""

    def set_domain(self, bounds: tuple[float, float] | None = None) -> None:
        """Sets the domain of the view.

        TODO: What is the "extent of the data"? Is it the bounds of the
        histogram, or the bounds of (clims + histogram)?

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the x axis to the passed
            values. If None, sets them to the extent of the data instead.
        """

    def set_range(self, bounds: tuple[float, float] | None = None) -> None:
        """Sets the range of the view.

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the y axis to the passed
            values. If None, sets them to the extent of the data instead.
        """

    def set_vertical(self, vertical: bool) -> None:
        """Sets the axis of the domain.

        Properties
        ----------
        vertical : bool
            If true, views the domain along the y axis and the range along the x
            axis. If false, views the domain along the x axis and the range along
            the y axis.
        """

    def set_range_log(self, enabled: bool) -> None:
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


class CursorType(Enum):
    DEFAULT = "default"
    V_ARROW = "v_arrow"
    H_ARROW = "h_arrow"
    ALL_ARROW = "all_arrow"
    BDIAG_ARROW = "bdiag_arrow"
    FDIAG_ARROW = "fdiag_arrow"

    def to_qt(self) -> Qt.CursorShape:
        """Converts CursorType to Qt.CursorShape."""
        from qtpy.QtCore import Qt

        return {
            CursorType.DEFAULT: Qt.CursorShape.ArrowCursor,
            CursorType.V_ARROW: Qt.CursorShape.SizeVerCursor,
            CursorType.H_ARROW: Qt.CursorShape.SizeHorCursor,
            CursorType.ALL_ARROW: Qt.CursorShape.SizeAllCursor,
            CursorType.BDIAG_ARROW: Qt.CursorShape.SizeBDiagCursor,
            CursorType.FDIAG_ARROW: Qt.CursorShape.SizeFDiagCursor,
        }[self]
