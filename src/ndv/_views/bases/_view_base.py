from abc import ABC, abstractmethod
from typing import Any


class Viewable(ABC):
    """ABC representing anything that can be viewed on screen.

    For example, a widget, a window, a frame, canvas, etc.
    """

    @abstractmethod
    def frontend_widget(self) -> Any:
        """Return the native object backing the viewable objects."""

    @abstractmethod
    def set_visible(self, visible: bool) -> None:
        """Sets the visibility of the view/widget itself."""

    @abstractmethod
    def close(self) -> None:
        """Close the view/widget."""
