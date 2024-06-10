from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

if TYPE_CHECKING:
    import cmap
    import numpy as np
    from qtpy.QtWidgets import QWidget


class PImageHandle(Protocol):
    @property
    def data(self) -> np.ndarray: ...
    @data.setter
    def data(self, data: np.ndarray) -> None: ...
    @property
    def visible(self) -> bool: ...
    @visible.setter
    def visible(self, visible: bool) -> None: ...
    @property
    def clim(self) -> Any: ...
    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None: ...
    @property
    def cmap(self) -> Any: ...
    @cmap.setter
    def cmap(self, cmap: Any) -> None: ...
    def remove(self) -> None: ...


class PCanvas(Protocol):
    def __init__(self, set_info: Callable[[str], None]) -> None: ...
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
        offset: tuple[float, float] | None = None,  # (Y, X)
    ) -> PImageHandle: ...
    def add_volume(
        self,
        data: np.ndarray | None = ...,
        cmap: cmap.Colormap | None = ...,
        offset: tuple[float, float, float] | None = ...,  # (Z, Y, X)
    ) -> PImageHandle: ...
