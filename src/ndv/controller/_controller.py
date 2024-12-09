from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ndv.models import DataDisplayModel
from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import LUTModel
from ndv.views import (
    get_canvas_class,
    get_histogram_canvas_class,
    get_histogram_frontend_class,
    get_view_frontend_class,
)

if TYPE_CHECKING:
    import cmap
    import numpy as np
    from typing_extensions import TypeAlias

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModel
    from ndv.models._stats import Stats
    from ndv.views.protocols import (
        PHistogramCanvas,
        PHistogramView,
        PImageHandle,
        PLutView,
        PView,
    )

    LutKey: TypeAlias = int | None


# (probably rename to something like just Viewer...
# or compose all three of the model/view/controller into a single class
# that exposes the public interface)


class ViewerController:
    """The controller mostly manages the connection between the model and the view.

    This is the primary public interface for the viewer.
    """

    def __init__(self, data: DataDisplayModel | None = None) -> None:
        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[LutKey, ChannelController] = {}

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
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)

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
            (model.events.channel_mode, self._on_model_channel_mode_changed),
            # TODO: lut values themselves are mutable evented objects...
            # so we need to connect to their events as well
            # (model.luts.value_changed, ...),
        ]:
            getattr(obj, _connect)(callback)

    # ------------------ Model callbacks ------------------

    def _fully_synchronize_view(self) -> None:
        """Fully re-synchronize the view with the model."""
        self._view.create_sliders(self._dd_model.canonical_data_coords)
        self._view.set_channel_mode(self.model.channel_mode)
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

    def _on_model_channel_mode_changed(self, mode: ChannelMode) -> None:
        self._view.set_channel_mode(mode)
        self._update_visible_sliders()
        show_channel_luts = mode in {ChannelMode.COLOR, ChannelMode.COMPOSITE}
        for lut_ctrl in self._lut_controllers.values():
            lut_ctrl.lut_view.setVisible(
                not show_channel_luts if lut_ctrl.key is None else show_channel_luts
            )
        # redraw
        self._clear_canvas()
        self._update_canvas()

    def _clear_canvas(self) -> None:
        for lut_ctrl in self._lut_controllers.values():
            # self._view.remove_lut_view(lut_ctrl.lut_view)
            while lut_ctrl.handles:
                lut_ctrl.handles.pop().remove()
        # do we need to cleanup the lut views themselves?

    # ------------------ View callbacks ------------------

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self.model.current_index.update(self._view.current_index())

    def _on_view_reset_zoom_clicked(self) -> None:
        """Reset the zoom level of the canvas."""
        self._canvas.set_range()

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

    def _on_view_channel_mode_changed(self, mode: ChannelMode) -> None:
        self.model.channel_mode = mode

    # ------------------ Helper methods ------------------

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        hidden_sliders = self._dd_model.canonical_visible_axes
        if self.model.channel_mode.is_multichannel():
            if ch := self._dd_model.canonical_channel_axis:
                hidden_sliders += (ch,)

        self._view.hide_sliders(hidden_sliders, show_remainder=True)

    def _update_canvas(self) -> None:
        """Force the canvas to fetch and update the displayed data.

        This is called (frequently) when anything changes that requires a redraw.
        It fetches the current data slice from the model and updates the image handle.
        """
        if not self._dd_model.data_wrapper:
            return

        # TODO: make asynchronous
        for future in self._dd_model.request_sliced_data():
            response = future.result()
            key = response.channel_key
            data = response.data

            if (lut_ctrl := self._lut_controllers.get(key)) is None:
                if key is None:
                    model = self.model.default_lut
                elif key in self.model.luts:
                    model = self.model.luts[key]
                else:
                    # we received a new channel key that has not been set in the model
                    # so we create a new LUT model for it
                    model = self.model.luts[key] = LUTModel()

                self._lut_controllers[key] = lut_ctrl = ChannelController(
                    key=key, view=self._view.add_lut_view(), model=model
                )

            if not lut_ctrl.handles:
                # we don't yet have any handles for this channel
                lut_ctrl.add_handle(self._canvas.add_image(data))
                # self._canvas.set_range()
            else:
                lut_ctrl.update_texture_data(data)

        self._canvas.refresh()

    def _get_values_at_world_point(self, x: int, y: int) -> dict[LutKey, float]:
        # TODO: handle 3D data
        if (x < 0 or y < 0) or self.model.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[LutKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((y, x))) is not None:
                values[key] = value

        return values

    def show(self) -> None:
        """Show the viewer."""
        self._view.show()


class HistogramController:
    """Manages the connection between a LUTModel, statistics and a histogram view."""

    def __init__(
        self,
        *,
        lut: LUTModel | None = None,
        stats: Stats | None = None,
    ) -> None:
        """Initializes a HistogramController.

        Properties
        ----------
        lut : LUTModel | None
            An initial LUTModel to attach.
        stats : Stats | None
            Initial statistics for display
        """
        # Canvas backend
        self._hist: PHistogramCanvas = get_histogram_canvas_class()()
        self._hist.climsChanged.connect(self._on_view_clims_update)
        self._hist.gammaChanged.connect(self._on_view_gamma_update)
        # Widget frontend
        self._view: PHistogramView = get_histogram_frontend_class()(self._hist)

        # Set initial statistics
        self._stats: Stats | None = None
        if stats:
            self.stats = stats
        # Set initial lut
        self._lut: LUTModel | None = None
        self.lut = lut

    @property
    def stats(self) -> Stats:
        """Return stats being displayed."""
        if self._stats is None:
            raise ValueError("Statistics have not yet been set!")
        return self._stats

    @stats.setter
    def stats(self, stats: Stats) -> None:
        """Set stats for display."""
        self._stats = stats
        self._hist.set_stats(stats)

    @property
    def lut(self) -> LUTModel | None:
        """Return the LUTModel currently attached."""
        return self._lut

    @lut.setter
    def lut(self, lut: LUTModel | None) -> None:
        """Sets the attached LUTModel."""
        connections = [
            ("clims", self._on_model_clims_update),
            ("cmap", self._on_model_cmap_update),
            ("gamma", self._on_model_gamma_update),
            ("visible", self._on_model_visible_update),
        ]
        old = self._lut
        self._lut = lut

        for signal_name, slot in connections:
            if old is not None:
                # Detach old LUT
                getattr(old.events, signal_name).disconnect(slot)
            if self._lut is not None:
                # Attach new LUT
                getattr(self._lut.events, signal_name).connect(slot)
            # Synchronize histogram with new LUT
            slot()

    def show(self) -> None:
        """Show the viewer."""
        self._view.show()

    # -- Private helpers -- #

    def _on_model_cmap_update(self) -> None:
        """Runs when the model's cmap changes."""
        # FIXME: Discrepancy between LUTModel and LUTView
        if self._lut:
            self._hist.set_colormap(self._lut.cmap)

    def _on_model_clims_update(self) -> None:
        """Runs when the model's clims change."""
        # FIXME: Discrepancy between LUTModel and LUTView
        if self._lut and (clims := self._lut.clims):
            self._hist.set_clims(clims)

    def _on_model_gamma_update(self) -> None:
        """Runs when the model's gamma changes."""
        if self._lut:
            self._hist.set_gamma(self._lut.gamma)

    def _on_model_visible_update(self) -> None:
        """Runs when the model's visibility changes."""
        visible = False if self._lut is None else self._lut.visible
        self._hist.set_lut_visible(visible)

    def _on_view_clims_update(self, clims: tuple[float, float]) -> None:
        """Runs when the user updates the clims on the view."""
        if self._lut is not None:
            self._lut.clims = clims

    def _on_view_gamma_update(self, gamma: float) -> None:
        """Runs when the user updates the gamma on the view."""
        if self._lut is not None:
            self._lut.gamma = gamma


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
