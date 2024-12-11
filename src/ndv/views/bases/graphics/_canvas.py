from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from ndv.views.bases._mouseable import Mouseable
from ndv.views.bases._view_base import Viewable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap
    import numpy as np

    from ndv._types import CursorType


class CanvasElement(Mouseable):
    """Protocol defining an interactive element on the Canvas."""

    @abstractmethod
    def visible(self) -> bool:
        """Defines whether the element is visible on the canvas."""

    @abstractmethod
    def set_visible(self, visible: bool) -> None:
        """Sets element visibility."""

    @abstractmethod
    def can_select(self) -> bool:
        """Defines whether the element can be selected."""

    @abstractmethod
    def selected(self) -> bool:
        """Returns element selection status."""

    @abstractmethod
    def set_selected(self, selected: bool) -> None:
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


class GraphicsCanvas(Viewable, Mouseable):
    @abstractmethod
    def refresh(self) -> None: ...
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


class ImageHandle(CanvasElement):
    @abstractmethod
    def data(self) -> np.ndarray: ...
    @abstractmethod
    def set_data(self, data: np.ndarray) -> None: ...
    @abstractmethod
    def clim(self) -> Any: ...
    @abstractmethod
    def set_clim(self, clims: tuple[float, float]) -> None: ...
    @abstractmethod
    def gamma(self) -> float: ...
    @abstractmethod
    def set_gamma(self, gamma: float) -> None: ...
    @abstractmethod
    def cmap(self) -> cmap.Colormap: ...
    @abstractmethod
    def set_cmap(self, cmap: cmap.Colormap) -> None: ...


class ArrayCanvas(GraphicsCanvas):
    @abstractmethod
    def set_ndim(self, ndim: Literal[2, 3]) -> None: ...
    @abstractmethod
    @abstractmethod
    def add_image(
        self,
        data: np.ndarray | None = ...,
        cmap: cmap.Colormap | None = ...,
        clims: tuple[float, float] | None = ...,
    ) -> ImageHandle: ...
    @abstractmethod
    def add_volume(
        self, data: np.ndarray | None = ..., cmap: cmap.Colormap | None = ...
    ) -> ImageHandle: ...

    @abstractmethod
    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> RoiHandle: ...
