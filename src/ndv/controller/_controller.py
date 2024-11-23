from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ndv.models import DataDisplayModel
from ndv.views import get_canvas_class, get_view_frontend_class

if TYPE_CHECKING:
    import cmap

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModel
    from ndv.views.protocols import PImageHandle, PLutView, PView


class ViewerController:
    """The controller mostly manages the connection between the model and the view."""

    def __init__(self, data: DataDisplayModel | None = None) -> None:
        if data is None:
            data = DataDisplayModel()

        # mapping of channel/LUT index to image handle, where None is the default LUT
        self._img_handles: defaultdict[int | None, list[PImageHandle]] = defaultdict(
            list
        )

        self._canvas = get_canvas_class()()
        self._canvas.set_ndim(2)

        view = get_view_frontend_class()(self._canvas.qwidget())

        # TODO: _dd_model is perhaps a temporary concept, and definitely name
        self._dd_model = data
        self._view = view

        self._set_model_connected(self._dd_model.display)
        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.mouseMoved.connect(self._on_view_mouse_moved)

        self._lut_views: dict[int | None, PLutView] = {}
        self._add_lut_view(None)

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
        self._view.create_sliders(self._dd_model.canonical_data_coords)
        if data is not None:
            self._update_visible_sliders()
            self._update_canvas()
            if wrapper := self._dd_model.data_wrapper:
                self._view.set_data_info(wrapper.summary_info())

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
            (model.events.visible_axes, self._on_model_visible_axes_changed),
            # the current_index attribute itself is immutable
            (model.current_index.value_changed, self._on_model_current_index_changed),
            # (model.events.channel_axis, self._on_channel_axis_changed),
            # TODO: lut values themselves are mutable evented objects...
            # so we need to connect to their events as well
            # (model.luts.value_changed, self._on_model_default_lut_cmap_changed),
            (model.default_lut.events.cmap, self._on_model_default_lut_cmap_changed),
            (model.default_lut.events.clims, self._on_model_default_lut_clims_changed),
        ]:
            getattr(obj, _connect)(callback)

    # ------------------ Model callbacks ------------------

    def _on_model_visible_axes_changed(self) -> None:
        self._update_visible_sliders()
        self._update_canvas()

    def _on_model_current_index_changed(self) -> None:
        value = self.model.current_index
        self._view.set_current_index(value)
        self._update_canvas()

    def _on_model_default_lut_cmap_changed(self) -> None:
        self._update_canvas()

    def _on_model_default_lut_clims_changed(self, clims: tuple[float, float]) -> None:
        for handle in self._img_handles[None]:
            handle.clim = clims

    # ------------------ View callbacks ------------------

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self.model.current_index.update(self._view.current_index())

    def _on_view_reset_zoom_clicked(self) -> None:
        """Reset the zoom level of the canvas."""
        self._canvas.set_range()

    def _on_view_lut_visible_changed(
        self, visible: bool, key: int | None = None
    ) -> None:
        for handle in self._img_handles[None]:
            handle.visible = visible

    def _on_view_lut_autoscale_changed(
        self, autoscale: bool, key: int | None = None
    ) -> None:
        self._dd_model.display.default_lut.autoscale = autoscale
        self._lut_views[key].setAutoScale(autoscale)

        if autoscale:
            lut = self.model.default_lut if key is None else self.model.luts[key]
            # TODO: or should we have a global min/max across all handles for this key?
            for handle in self._img_handles[key]:
                data = handle.data
                lut.clims = (data.min(), data.max())
                # self._handles[None].clim = (data.min(), data.max())
                # self._lut_views[None].setClims((data.min(), data.max()))

    def _on_view_lut_cmap_changed(
        self, cmap: cmap.Colormap, key: int | None = None
    ) -> None:
        lut = self.model.default_lut if key is None else self.model.luts[key]
        for handle in self._img_handles[key]:
            handle.cmap = cmap
            lut.cmap = cmap

    def _on_view_lut_clims_changed(self, clims: tuple[float, float]) -> None:
        self.model.default_lut.clims = clims
        self.model.default_lut.autoscale = False

    # ------------------ Helper methods ------------------

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        self._view.hide_sliders(
            self._dd_model.canonical_visible_axes, show_remainder=True
        )

    def _update_canvas(self) -> None:
        """Force the canvas to fetch and update the displayed data.

        This is called (frequently) when anything changes that requires a redraw.
        It fetches the current data slice from the model and updates the image handle.
        """
        if not self._dd_model.data_wrapper:
            return

        data = self._dd_model.current_data_slice()  # TODO: make asynchronous
        if None in self._img_handles:
            if handles := self._img_handles[None]:
                # until we have a more sophisticated way to handle updating data for
                # multiple handles, we'll just update the first one
                handles[0].data = data
                # if this image handle is visible and autoscale is on, then we need to
                # update the clim values
                if self.model.default_lut.autoscale:
                    self.model.default_lut.clims = (data.min(), data.max())
                    # self._lut_views[None].setClims((data.min(), data.max()))
                    # technically... the LutView may also emit a signal that the
                    # controller listens to, and then updates the image handle
                    # but this next line is more direct
                    # self._handles[None].clim = (data.min(), data.max())
        else:
            handle = self._canvas.add_image(data)
            self._img_handles[None].append(handle)
            self._canvas.set_range()
            handle.cmap = self.model.default_lut.cmap
            if clims := self.model.default_lut.clims:
                handle.clim = clims
        self._canvas.refresh()

    def _add_lut_view(self, key: int | None) -> PLutView:
        if key in self._lut_views:
            # need to clean up
            raise NotImplementedError(f"LUT view with key {key} already exists")
        self._lut_views[key] = lut = self._view.add_lut_view()

        lut.visibleChanged.connect(self._on_view_lut_visible_changed)
        lut.autoscaleChanged.connect(self._on_view_lut_autoscale_changed)
        lut.cmapChanged.connect(self._on_view_lut_cmap_changed)
        lut.climsChanged.connect(self._on_view_lut_clims_changed)

        model_lut = self._dd_model.display.default_lut
        model_lut.events.cmap.connect(lut.setColormap)
        model_lut.events.clims.connect(lut.setClims)
        model_lut.events.autoscale.connect(lut.setAutoScale)
        model_lut.events.visible.connect(lut.setLutVisible)
        return lut

    def _on_view_mouse_moved(self, event: MouseMoveEvent) -> None:
        """Update text of hover_info_label with data value(s) at point."""
        x, y, _z = self._canvas.canvas_to_world((event.x, event.y))
        channel_values = self._get_values_at_world_point(int(x), int(y))
        vals = []
        for ch, value in channel_values.items():
            # restrict to 2 decimal places, but remove trailing zeros
            fval = f"{value:.2f}".rstrip("0").rstrip(".")
            vals.append(f"{ch}: {fval}")
        text = f"[{y:.1f}, {x:.1f}]" + ",".join(vals)
        self._view.set_hover_info(text)

    def _get_values_at_world_point(self, x: int, y: int) -> dict[int | None, float]:
        # TODO: handle 3D data
        if (x < 0 or y < 0) or self.model.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[int | None, float] = {}
        for channel_key, handles in enumerate(self._img_handles.values()):
            if not handles:
                continue
            # only getting one handle per channel for now
            handle = handles[0]
            with suppress(IndexError):  # skip out of bounds
                # here, we're retrieving the value from the in-memory data
                # stored by the backend visual, rather than querying the data itself
                # this is a quick workaround to get the value without having to
                # worry about higher dimensions in the data source (since the
                # texture has already been reduced to 2D). But a more complete
                # implementation would gather the full current nD index and query
                # the data source directly.
                values[channel_key] = handle.data[y, x]

        return values
