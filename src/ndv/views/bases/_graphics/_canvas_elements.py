from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ._mouseable import Mouseable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap as _cmap
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


class ImageHandle(CanvasElement):
    @abstractmethod
    def data(self) -> np.ndarray: ...
    @abstractmethod
    def set_data(self, data: np.ndarray) -> None: ...
    @abstractmethod
    def clims(self) -> tuple[float, float]: ...
    @abstractmethod
    def set_clims(self, clims: tuple[float, float]) -> None: ...
    @abstractmethod
    def gamma(self) -> float: ...
    @abstractmethod
    def set_gamma(self, gamma: float) -> None: ...
    @abstractmethod
    def cmap(self) -> _cmap.Colormap: ...
    @abstractmethod
    def set_cmap(self, cmap: _cmap.Colormap) -> None: ...


class RoiHandle(CanvasElement):
    @abstractmethod
    def vertices(self) -> Sequence[Sequence[float]]: ...
    @abstractmethod
    def set_vertices(self, data: Sequence[Sequence[float]]) -> None: ...
    @abstractmethod
    def color(self) -> Any: ...
    @abstractmethod
    def set_color(self, color: _cmap.Color | None) -> None: ...
    @abstractmethod
    def border_color(self) -> Any: ...
    @abstractmethod
    def set_border_color(self, color: _cmap.Color | None) -> None: ...
