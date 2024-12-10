from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Union

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap
    import numpy as np
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QWidget

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
    visibleChanged: PSignal
    autoscaleChanged: PSignal
    cmapChanged: PSignal
    climsChanged: PSignal

    def set_name(self, name: str) -> None: ...
    def set_auto_scale(self, auto: bool) -> None: ...
    def set_colormap(self, cmap: cmap.Colormap) -> None: ...
    def set_clims(self, clims: tuple[float, float]) -> None: ...
    def set_lut_visible(self, visible: bool) -> None: ...
    def setVisible(self, visible: bool) -> None: ...


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
    mouseMoved: PSignal  # Signal(_types.MouseMoveEvent)
    channelModeChanged: PSignal

    def __init__(self, canvas_widget: Any, **kwargs: Any) -> None: ...
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


class Mouseable(Protocol):
    def on_mouse_move(self, event: MouseMoveEvent) -> bool: ...
    def on_mouse_press(self, event: MousePressEvent) -> bool: ...
    def on_mouse_release(self, event: MouseReleaseEvent) -> bool: ...


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
    def qwidget(self) -> QWidget: ...
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
