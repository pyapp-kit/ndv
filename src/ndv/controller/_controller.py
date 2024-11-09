from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ndv.models import DataDisplayModel
from ndv.views import get_view_backend

if TYPE_CHECKING:
    import cmap

    from ndv.models._array_display_model import ArrayDisplayModel
    from ndv.views.protocols import PImageHandle, PLutView, PView


class ViewerController:
    """The controller mostly manages the connection between the model and the view."""

    def __init__(
        self, view: PView | None = None, data: DataDisplayModel | None = None
    ) -> None:
        if data is None:
            data = DataDisplayModel()
        if view is None:
            view = get_view_backend()
        self._dd_model = data  # rename me?
        self._view = view

        self._set_model_connected(self._dd_model.display)
        self._view.currentIndexChanged.connect(self._on_slider_value_changed)
        self._handles: dict[int | None, PImageHandle] = {}

        self._lut_views: dict[int | None, PLutView] = {}
        self.add_lut_view(None)

    # -------------- possibly move this logic up to DataDisplayModel --------------
    @property
    def view(self) -> PView:
        """Return the front-end view object."""
        return self._view

    @property
    def model(self) -> ArrayDisplayModel:
        """Return the display model for the viewer."""
        return self._dd_model.display

    @model.setter
    def model(self, display_model: ArrayDisplayModel) -> None:
        """Set the display model for the viewer."""
        previous_model, self._dd_model.display = self._dd_model.display, display_model
        self._set_model_connected(previous_model, False)
        self._set_model_connected(display_model)

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._dd_model.data_wrapper is None:
            return None
        # returning the actual data, not the wrapper
        return self._dd_model.data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        self._dd_model.data = data
        print("Creating sliders", self._dd_model.canonical_data_coords)
        self._view.create_sliders(self._dd_model.canonical_data_coords)
        if data is not None:
            self._update_visible_sliders()
            self._update_canvas()

    # -----------------------------------------------------------------------------

    def _set_model_connected(
        self, model: ArrayDisplayModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.  (but it's kinda ugly)
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.visible_axes, self._on_visible_axes_changed),
            # the current_index attribute itself is immutable
            (model.current_index.value_changed, self._on_current_index_changed),
            # (model.events.channel_axis, self._on_channel_axis_changed),
            # TODO: lut values themselves are mutable evented objects...
            # so we need to connect to their events as well
            (model.luts.value_changed, self._on_luts_changed),
            (model.default_lut.events.cmap, self._on_luts_changed),
        ]:
            getattr(obj, _connect)(callback)

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        self._view.hide_sliders(
            self._dd_model.canonical_visible_axes, show_remainder=True
        )

    def _on_current_index_changed(self) -> None:
        value = self.model.current_index
        self._view.set_current_index(value)
        self._update_canvas()

    def _on_slider_value_changed(self) -> None:
        """Update the model when slider value changes."""
        self.model.current_index.update(self._view.current_index())

    def _update_canvas(self) -> None:
        if not self._dd_model.data_wrapper:
            return
        data = self._dd_model.current_data_slice()  # TODO: make asynchronous
        if None in self._handles:
            self._handles[None].data = data
        else:
            self._handles[None] = self._view.add_image_to_canvas(data)
            self._handles[None].cmap = self.model.default_lut.cmap
            if clims := self.model.default_lut.clims:
                self._handles[None].clim = clims
        self._view.refresh()

    def _on_luts_changed(self) -> None:
        self._update_canvas()

    def _on_visible_axes_changed(self) -> None:
        self._update_visible_sliders()
        self._update_canvas()

    def add_lut_view(self, key: int | None) -> PLutView:
        if key in self._lut_views:
            # need to clean up
            raise NotImplementedError(f"LUT view with key {key} already exists")
        self._lut_views[key] = lut = self._view.add_lut_view()

        lut.visibleChanged.connect(self._on_lut_visible_changed)
        lut.autoscaleChanged.connect(self._on_autoscale_changed)
        lut.cmapChanged.connect(self._on_cmap_changed)
        lut.climsChanged.connect(self._on_clims_changed)

        model_lut = self._dd_model.display.default_lut
        model_lut.events.cmap.connect(lut.setColormap)
        model_lut.events.clims.connect(lut.setClims)
        model_lut.events.autoscale.connect(lut.setAutoScale)
        model_lut.events.visible.connect(lut.setLutVisible)
        return lut

    def _on_lut_visible_changed(self, visible: bool) -> None:
        self._handles[None].visible = visible

    def _on_autoscale_changed(self, autoscale: bool) -> None: ...

    def _on_cmap_changed(self, cmap: cmap.Colormap) -> None:
        self._handles[None].cmap = cmap

    def _on_clims_changed(self, clims: tuple[float, float]) -> None:
        self._handles[None].clim = clims
