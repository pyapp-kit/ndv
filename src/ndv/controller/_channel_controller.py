from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cmap
    import numpy as np

    from ndv.models._lut_model import LUTModel
    from ndv.views.protocols import PImageHandle, PLutView

    LutKey = int | None


class ChannelController:
    """Controller for a single channel in the viewer.

    This manages the connection between the LUT model (settings like colormap,
    contrast limits and visibility) and the LUT view (the front-end widget that
    allows the user to interact with these settings), as well as the image handle
    that displays the data, all for a single "channel" extracted from the data.
    """

    def __init__(self, key: LutKey, view: PLutView, model: LUTModel) -> None:
        self.key = key
        self.lut_view = view
        self.lut_model = model
        self.handles: list[PImageHandle] = []

        # setup the initial state of the LUT view
        self._update_view_from_model()

        # connect view changes to controller callbacks that update the model
        self.lut_view.visibleChanged.connect(self._on_view_lut_visible_changed)
        self.lut_view.autoscaleChanged.connect(self._on_view_lut_autoscale_changed)
        self.lut_view.cmapChanged.connect(self._on_view_lut_cmap_changed)
        self.lut_view.climsChanged.connect(self._on_view_lut_clims_changed)

        # connect model changes to view callbacks that update the view
        self.lut_model.events.cmap.connect(self._on_model_cmap_changed)
        self.lut_model.events.clims.connect(self._on_model_clims_changed)
        self.lut_model.events.autoscale.connect(view.set_auto_scale)
        self.lut_model.events.visible.connect(self._on_model_visible_changed)

    def _on_model_clims_changed(self, clims: tuple[float, float]) -> None:
        """The contrast limits in the model have changed."""
        self.lut_view.set_clims(clims)
        for handle in self.handles:
            handle.clim = clims

    def _on_model_cmap_changed(self, cmap: cmap.Colormap) -> None:
        """The colormap in the model has changed."""
        self.lut_view.set_colormap(cmap)
        for handle in self.handles:
            handle.cmap = cmap

    def _on_model_visible_changed(self, visible: bool) -> None:
        """The visibility in the model has changed."""
        self.lut_view.set_lut_visible(visible)
        for handle in self.handles:
            handle.visible = visible

    def _update_view_from_model(self) -> None:
        """Make sure the view matches the model."""
        self.lut_view.set_colormap(self.lut_model.cmap)
        if self.lut_model.clims:
            self.lut_view.set_clims(self.lut_model.clims)
        # TODO: handle more complex autoscale types
        self.lut_view.set_auto_scale(bool(self.lut_model.autoscale))
        self.lut_view.set_lut_visible(True)
        name = str(self.key) if self.key is not None else ""
        self.lut_view.set_name(name)

    def _on_view_lut_visible_changed(self, visible: bool, key: LutKey = None) -> None:
        """The visibility checkbox in the LUT widget has changed."""
        for handle in self.handles:
            handle.visible = visible

    def _on_view_lut_autoscale_changed(
        self, autoscale: bool, key: LutKey = None
    ) -> None:
        """The autoscale checkbox in the LUT widget has changed."""
        self.lut_model.autoscale = autoscale
        self.lut_view.set_auto_scale(autoscale)

        if autoscale:
            # TODO: or should we have a global min/max across all handles for this key?
            for handle in self.handles:
                data = handle.data
                # update the model with the new clim values
                self.lut_model.clims = (data.min(), data.max())

    def _on_view_lut_cmap_changed(
        self, cmap: cmap.Colormap, key: LutKey = None
    ) -> None:
        """The colormap in the LUT widget has changed."""
        for handle in self.handles:
            handle.cmap = cmap  # actually apply it to the Image texture
            self.lut_model.cmap = cmap  # update the model as well

    def _on_view_lut_clims_changed(self, clims: tuple[float, float]) -> None:
        """The contrast limits slider in the LUT widget has changed."""
        self.lut_model.clims = clims
        # when the clims are manually adjusted in the view, we turn off autoscale
        self.lut_model.autoscale = False

    def update_texture_data(self, data: np.ndarray) -> None:
        """Update the data in the image handle."""
        # WIP:
        # until we have a more sophisticated way to handle updating data
        # for multiple handles, we'll just update the first one
        if not (handles := self.handles):
            return
        handles[0].data = data
        # if this image handle is visible and autoscale is on, then we need
        # to update the clim values
        if self.lut_model.autoscale:
            self.lut_model.clims = (data.min(), data.max())
            # lut_view.setClims((data.min(), data.max()))
            # technically... the LutView may also emit a signal that the
            # controller listens to, and then updates the image handle
            # but this next line is more direct
            # self._handles[None].clim = (data.min(), data.max())

    def add_handle(self, handle: PImageHandle) -> None:
        """Add an image texture handle to the controller."""
        self.handles.append(handle)
        handle.cmap = self.lut_model.cmap
        if self.lut_model.autoscale:
            self.lut_model.clims = (handle.data.min(), handle.data.max())
        if self.lut_model.clims:
            handle.clim = self.lut_model.clims

    def get_value_at_index(self, idx: tuple[int, ...]) -> float | None:
        """Get the value of the data at the given index."""
        if not (handles := self.handles):
            return None
        # only getting one handle per channel for now
        handle = handles[0]
        with suppress(IndexError):  # skip out of bounds
            # here, we're retrieving the value from the in-memory data
            # stored by the backend visual, rather than querying the data itself
            # this is a quick workaround to get the value without having to
            # worry about other dimensions in the data source (since the
            # texture has already been reduced to 2D). But a more complete
            # implementation would gather the full current nD index and query
            # the data source directly.
            return handle.data[idx]  # type: ignore [no-any-return]
        return None
