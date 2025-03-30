from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._view_base import Viewable

if TYPE_CHECKING:
    import cmap

    from ndv.models._lut_model import ClimPolicy, LUTModel


class LutView(Viewable):
    """Manages LUT properties (contrast, colormap, etc...) in a view object."""

    _model: LUTModel | None = None

    @abstractmethod
    def set_channel_name(self, name: str) -> None:
        """Set the name of the channel to `name`."""

    @abstractmethod
    def set_clim_policy(self, policy: ClimPolicy) -> None:
        """Set the clim policy to `policy`.

        Usually corresponds to an "autoscale" checkbox.

        Note that this method must not modify the backing LUTModel.
        """

    @abstractmethod
    def set_colormap(self, cmap: cmap.Colormap) -> None:
        """Set the colormap to `cmap`.

        Usually corresponds to a dropdown menu.

        Note that this method must not modify the backing LUTModel.
        """

    @abstractmethod
    def set_clims(self, clims: tuple[float, float]) -> None:
        """Set the (low, high) contrast limits to `clims`.

        Usually this will be a range slider or two text boxes.

        Note that this method must not modify the backing LUTModel.
        """

    @abstractmethod
    def set_channel_visible(self, visible: bool) -> None:
        """Check or uncheck the visibility indicator of the LUT.

        Usually corresponds to a checkbox.

        Note that this method must not modify the backing LUTModel.
        """

    def set_gamma(self, gamma: float) -> None:
        """Set the gamma value of the LUT.

        Note that this method must not modify the backing LUTModel.
        """
        return None

    @property
    def model(self) -> LUTModel | None:
        return self._model

    @model.setter
    def model(self, model: LUTModel | None) -> None:
        # Disconnect old model
        if self._model is not None:
            self._model.events.clims.disconnect(self.set_clim_policy)
            self._model.events.cmap.disconnect(self.set_colormap)
            self._model.events.gamma.disconnect(self.set_gamma)
            self._model.events.visible.disconnect(self.set_channel_visible)

        # Connect new model
        self._model = model
        if self._model is not None:
            self._model.events.clims.connect(self.set_clim_policy)
            self._model.events.cmap.connect(self.set_colormap)
            self._model.events.gamma.connect(self.set_gamma)
            self._model.events.visible.connect(self.set_channel_visible)

        self.synchronize()

    def synchronize(self) -> None:
        """Aligns the view against the backing model."""
        if model := self._model:
            self.set_clim_policy(model.clims)
            self.set_colormap(model.cmap)
            self.set_gamma(model.gamma)
            self.set_channel_visible(model.visible)
