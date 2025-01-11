from abc import abstractmethod
from typing import final

import cmap
from psygnal import Signal

from ._view_base import Viewable


class LutView(Viewable):
    """Manages LUT properties (contrast, colormap, etc...) in a view object."""

    visibilityChanged = Signal(bool)
    autoscaleChanged = Signal(bool)
    cmapChanged = Signal(cmap.Colormap)
    climsChanged = Signal(tuple)
    gammaChanged = Signal(float)

    @abstractmethod
    def set_channel_name(self, name: str) -> None:
        """Set the name of the channel to `name`."""

    @abstractmethod
    def set_auto_scale(self, checked: bool) -> None:
        """Set the autoscale button to checked if `checked` is True."""

    @abstractmethod
    def set_colormap(self, cmap: cmap.Colormap) -> None:
        """Set the colormap to `cmap`.

        Usually corresponds to a dropdown menu.
        """

    @abstractmethod
    def set_clims(self, clims: tuple[float, float]) -> None:
        """Set the (low, high) contrast limits to `clims`.

        Usually this will be a range slider or two text boxes.
        """

    @abstractmethod
    def set_channel_visible(self, visible: bool) -> None:
        """Check or uncheck the visibility indicator of the LUT.

        Usually corresponds to a checkbox.
        """

    def set_gamma(self, gamma: float) -> None:
        """Set the gamma value of the LUT."""
        return None

    # These methods apply a value to the view without re-emitting the signal.

    @final
    def set_auto_scale_without_signal(self, auto: bool) -> None:
        with self.autoscaleChanged.blocked():
            self.set_auto_scale(auto)

    @final
    def set_colormap_without_signal(self, cmap: cmap.Colormap) -> None:
        with self.cmapChanged.blocked():
            self.set_colormap(cmap)

    @final
    def set_clims_without_signal(self, clims: tuple[float, float]) -> None:
        with self.climsChanged.blocked():
            self.set_clims(clims)

    @final
    def set_gamma_without_signal(self, gamma: float) -> None:
        with self.gammaChanged.blocked():
            self.set_gamma(gamma)

    @final
    def set_channel_visible_without_signal(self, visible: bool) -> None:
        with self.visibilityChanged.blocked():
            self.set_channel_visible(visible)
