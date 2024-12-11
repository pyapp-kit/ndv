from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from ndv.views.bases._mouseable import Mouseable
from ndv.views.bases._view_base import Viewable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap
    import numpy as np

    from ndv.views.bases.graphics._canvas_elements import (
        CanvasElement,
        ImageHandle,
        RoiHandle,
    )


class GraphicsCanvas(Viewable, Mouseable):
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


class ArrayCanvas(GraphicsCanvas):
    @abstractmethod
    def set_ndim(self, ndim: Literal[2, 3]) -> None: ...
    @abstractmethod
    @abstractmethod
    def add_image(self, data: np.ndarray | None = ...) -> ImageHandle: ...
    @abstractmethod
    def add_volume(self, data: np.ndarray | None = ...) -> ImageHandle: ...
    @abstractmethod
    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> RoiHandle: ...
