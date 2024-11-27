from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ndv.models import DataDisplayModel
from ndv.models._lut_model import LUTModel
from ndv.models._stats import Stats
from ndv.views import (
    get_canvas_class,
    get_histogram_backend_class,
    get_histogram_frontend_class,
    get_view_frontend_class,
)

if TYPE_CHECKING:
    import cmap

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModel
    from ndv.views.protocols import (
        PHistogramCanvas,
        PHistogramView,
        PImageHandle,
        PLutView,
        PView,
    )


# (probably rename to something like just Viewer)
class ViewerController:
    """The controller mostly manages the connection between the model and the view.

    This is the primary public interface for the viewer.
    """

    def __init__(self, data: DataDisplayModel | None = None) -> None:
        # mapping of channel/LUT index to image handle, where None is the default LUT
        # PImageHandle is an object that allows this controller to update the canvas img
        self._img_handles: defaultdict[int | None, list[PImageHandle]] = defaultdict(
            list
        )
        # mapping of channel/LUT index to LutView, where None is the default LUT
        # LutView is a front-end object that allows the user to interact with the LUT
        self._lut_views: dict[int | None, PLutView] = {}

        # get and create the front-end and canvas classes
        frontend_cls = get_view_frontend_class()
        canvas_cls = get_canvas_class()
        self._canvas = canvas_cls()
        self._canvas.set_ndim(2)
        self._view = frontend_cls(self._canvas.qwidget())

        # TODO: _dd_model is perhaps a temporary concept, and definitely name
        self._dd_model = data or DataDisplayModel()

        self._set_model_connected(self._dd_model.display)
        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.mouseMoved.connect(self._on_view_mouse_moved)

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
        self._fully_synchronize_view()

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
            (
                model.default_lut.events.visible,
                self._on_model_default_lut_visible_changed,
            ),
            (model.default_lut.events.cmap, self._on_model_default_lut_cmap_changed),
            (model.default_lut.events.clims, self._on_model_default_lut_clims_changed),
        ]:
            getattr(obj, _connect)(callback)

    # ------------------ Model callbacks ------------------

    def _fully_synchronize_view(self) -> None:
        """Fully re-synchronize the view with the model."""
        self._view.create_sliders(self._dd_model.canonical_data_coords)
        if self.data is not None:
            self._update_visible_sliders()
            # if we have data:
            if wrapper := self._dd_model.data_wrapper:
                self._view.set_data_info(wrapper.summary_info())

            self._update_canvas()

    def _on_model_visible_axes_changed(self) -> None:
        self._update_visible_sliders()
        self._update_canvas()

    def _on_model_current_index_changed(self) -> None:
        value = self.model.current_index
        self._view.set_current_index(value)
        self._update_canvas()

    def _on_model_default_lut_visible_changed(self, visible: bool) -> None:
        for handle in self._img_handles[None]:
            handle.visible = visible

    def _on_model_default_lut_cmap_changed(self) -> None:
        for handle in self._img_handles[None]:
            handle.cmap = self.model.default_lut.cmap

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
        lut_view = self._lut_views[key]
        lut_view.set_auto_scale(autoscale)

        if autoscale:
            lut_model = self.model.default_lut if key is None else self.model.luts[key]
            # TODO: or should we have a global min/max across all handles for this key?
            for handle in self._img_handles[key]:
                data = handle.data
                # update the model with the new clim values
                lut_model.clims = (data.min(), data.max())

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

        key = None  # TODO: handle multiple channels

        if key in self._lut_views:
            self._lut_views[key]
        else:
            self._add_lut_view(key)

        if key is None:
            lut_model = self.model.default_lut
        else:
            lut_model = self.model.luts[key]

        data = self._dd_model.current_data_slice()  # TODO: make asynchronous
        if key in self._img_handles:
            if handles := self._img_handles[key]:
                # until we have a more sophisticated way to handle updating data for
                # multiple handles, we'll just update the first one
                handles[0].data = data
                # if this image handle is visible and autoscale is on, then we need to
                # update the clim values
                if lut_model.autoscale:
                    lut_model.clims = (data.min(), data.max())
                    # lut_view.setClims((data.min(), data.max()))
                    # technically... the LutView may also emit a signal that the
                    # controller listens to, and then updates the image handle
                    # but this next line is more direct
                    # self._handles[None].clim = (data.min(), data.max())
        else:
            handle = self._canvas.add_image(data)
            self._img_handles[key].append(handle)
            self._canvas.set_range()
            handle.cmap = lut_model.cmap
            if clims := lut_model.clims:
                handle.clim = clims
        self._canvas.refresh()

    def _add_lut_view(self, key: int | None) -> PLutView:
        """Create a new LUT view and connect it to the model."""
        if key in self._lut_views:
            # need to clean up
            raise NotImplementedError(f"LUT view with key {key} already exists")

        self._lut_views[key] = lut_view = self._view.add_lut_view()
        lut_model = self.model.default_lut if key is None else self.model.luts[key]

        # setup the initial state of the LUT view
        lut_view.set_colormap(lut_model.cmap)
        if lut_model.clims:
            lut_view.set_clims(lut_model.clims)
        # TODO: handle more complex autoscale types
        lut_view.set_auto_scale(bool(lut_model.autoscale))
        lut_view.set_lut_visible(True)

        # connect view changes to controller callbacks that update the model
        lut_view.visibleChanged.connect(self._on_view_lut_visible_changed)
        lut_view.autoscaleChanged.connect(self._on_view_lut_autoscale_changed)
        lut_view.cmapChanged.connect(self._on_view_lut_cmap_changed)
        lut_view.climsChanged.connect(self._on_view_lut_clims_changed)

        # connect model changes to view callbacks that update the view
        lut_model.events.cmap.connect(lut_view.set_colormap)
        lut_model.events.clims.connect(lut_view.set_clims)
        lut_model.events.autoscale.connect(lut_view.set_auto_scale)
        lut_model.events.visible.connect(lut_view.set_lut_visible)
        return lut_view

    def _on_view_mouse_moved(self, event: MouseMoveEvent) -> None:
        """Respond to a mouse move event in the view."""
        x, y, _z = self._canvas.canvas_to_world((event.x, event.y))

        # collect and format intensity values at the current mouse position
        channel_values = self._get_values_at_world_point(int(x), int(y))
        vals = []
        for ch, value in channel_values.items():
            # restrict to 2 decimal places, but remove trailing zeros
            fval = f"{value:.2f}".rstrip("0").rstrip(".")
            fch = f"{ch}: " if ch is not None else ""
            vals.append(f"{fch}{fval}")
        text = f"[{y:.0f}, {x:.0f}] " + ",".join(vals)
        self._view.set_hover_info(text)

    def _get_values_at_world_point(self, x: int, y: int) -> dict[int | None, float]:
        # TODO: handle 3D data
        if (x < 0 or y < 0) or self.model.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[int | None, float] = {}
        for channel_key, handles in self._img_handles.items():
            if not handles:
                continue
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
                values[channel_key] = handle.data[y, x]

        return values

    def show(self) -> None:
        """Show the viewer."""
        self._view.show()


class HistogramController:
    """Manages the connection between models and a histogram view."""

    def __init__(
        self,
        data: DataDisplayModel | None = None,
        lut: LUTModel | None = None,
    ) -> None:
        self._lut = lut or LUTModel()
        self._data = data or DataDisplayModel()

        self._hist: PHistogramCanvas = get_histogram_backend_class()()
        self._view: PHistogramView = get_histogram_frontend_class()(self._hist)

        # A HistogramView is both a StatsView and a LUTView
        # DataDisplayModel <-> StatsView
        self._data.display.current_index.value_changed.connect(self._update_data)
        # # LutModel <-> LutView
        self._lut.events.clims.connect(self._set_model_clims)
        self._hist.climsChanged.connect(self._set_view_clims)
        self._lut.events.gamma.connect(self._set_model_gamma)
        self._hist.gammaChanged.connect(self._set_view_gamma)

    def _update_data(self) -> None:
        data = self._data.current_data_slice()  # TODO: make asynchronous
        stats = Stats(data=data)  # TODO: make asynchronous
        values, bin_edges = stats.histogram
        # TODO: Display average, min, max, std_deviation?
        self._hist.set_histogram(values, bin_edges)

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._data.data_wrapper is None:
            return None
        # returning the actual data, not the wrapper
        return self._data.data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        self._data.data = data
        self._update_data()

    def _set_model_clims(self) -> None:
        clims = self._lut.clims
        # FIXME: Discrepancy between LUTModel and LUTView
        if clims is not None:
            self._hist.set_clims(clims)

    def _set_view_clims(self, clims: tuple[float, float]) -> None:
        self._lut.clims = clims

    def _set_model_gamma(self) -> None:
        gamma = self._lut.gamma
        self._hist.set_gamma(gamma)

    def _set_view_gamma(self, gamma: float) -> None:
        self._lut.gamma = gamma

    def view(self) -> Any:
        """Returns an object that can be displayed by the active backend."""
        return self._view
