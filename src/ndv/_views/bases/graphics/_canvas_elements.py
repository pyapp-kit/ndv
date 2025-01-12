from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ndv._views.bases._lut_view import LutView

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


class ImageHandle(CanvasElement, LutView):
    @abstractmethod
    def data(self) -> np.ndarray: ...
    @abstractmethod
    def set_data(self, data: np.ndarray) -> None: ...
    @abstractmethod
    def clim(self) -> Any: ...
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

    # -- LutView methods -- #
    def close(self) -> None:
        pass

    def frontend_widget(self) -> Any:
        return None

    def set_channel_name(self, name: str) -> None:
        pass

    def set_auto_scale(self, checked: bool) -> None:
        # TODO: Make this computation alter the slider...
        if checked:
            d = self.data()
            self.set_clims((d.min(), d.max()))

    def set_colormap(self, cmap: _cmap.Colormap) -> None:
        self.set_cmap(cmap)

    def set_channel_visible(self, visible: bool) -> None:
        self.set_visible(visible)

    # set_clims, set_gamma reused above


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
