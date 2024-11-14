from __future__ import annotations

from abc import abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from ._mouseable import Mouseable

if TYPE_CHECKING:
    import cmap as _cmap
    import numpy as np


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


class ROIMoveMode(Enum):
    """Describes graphical mechanisms for ROI translation."""

    HANDLE = auto()  # Moving one handle (but not all)
    TRANSLATE = auto()  # Translating everything


class RectangularROI(CanvasElement):
    """An axis-aligned rectanglular ROI."""

    boundingBoxChanged = Signal(tuple[tuple[float, float], tuple[float, float]])

    def set_bounding_box(
        self, minimum: tuple[float, float], maximum: tuple[float, float]
    ) -> None:
        """Sets the bounding box."""

    def set_fill(self, color: _cmap.Color) -> None:
        """Sets the fill color."""

    def set_border(self, color: _cmap.Color) -> None:
        """Sets the border color."""

    def set_handles(self, color: _cmap.Color) -> None:
        """Sets the handle face color."""
