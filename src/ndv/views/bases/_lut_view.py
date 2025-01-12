from abc import abstractmethod

import cmap

from ndv.models._lut_model import LUTModel

from ._view_base import Viewable


class LutView(Viewable):
    """Manages LUT properties (contrast, colormap, etc...) in a view object."""

    _model: LUTModel | None = None

    def __init__(self, model: LUTModel | None = None) -> None:
        self.model = model

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

    @property
    def model(self) -> LUTModel | None:
        return self._model

    @model.setter
    def model(self, model: LUTModel | None) -> None:
        if self._model is not None:
            self._model.events.autoscale.disconnect(self.set_auto_scale)
            self._model.events.clims.disconnect(self.set_clims)
            self._model.events.cmap.disconnect(self.set_colormap)
            self._model.events.gamma.disconnect(self.set_gamma)
            self._model.events.visible.disconnect(self.set_channel_visible)
        self._model = model
        if self._model is not None:
            self._model.events.autoscale.connect(self.set_auto_scale)
            self._model.events.clims.connect(self.set_clims)
            self._model.events.cmap.connect(self.set_colormap)
            self._model.events.gamma.connect(self.set_gamma)
            self._model.events.visible.connect(self.set_channel_visible)

        self.synchronize()

    def synchronize(self) -> None:
        if model := self._model:
            self.set_auto_scale(bool(model.autoscale))
            if model.clims:
                self.set_clims(model.clims)
            self.set_colormap(model.cmap)
            self.set_gamma(model.gamma)
            self.set_channel_visible(model.visible)
