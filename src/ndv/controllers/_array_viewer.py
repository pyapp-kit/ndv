from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import ArrayDataDisplayModel
from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._lut_model import LUTModel
from ndv.views import _app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from typing_extensions import TypeAlias

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModelKwargs
    from ndv.models.data_wrappers import DataWrapper
    from ndv.views.bases import ArrayView, HistogramCanvas

    LutKey: TypeAlias = int | None


# primary "Controller" (and public API) for viewing an array


class ArrayViewer:
    """Viewer dedicated to displaying a single n-dimensional array.

    This wraps a model, view, and controller into a single object, and defines the
    public API.

    Parameters
    ----------
    data_or_model : ArrayDataDisplayModel | DataWrapper | Any
        Data to be displayed. If a full `ArrayDataDisplayModel` is provided, it will be
        used directly. If an array or `DataWrapper` is provided, a default display model
        will be created.
    display_model : ArrayDisplayModel, optional
        Just the display model to use. If provided, `data_or_model` must be an array
        or `DataWrapper`... and kwargs will be ignored.
    **kwargs: ArrayDisplayModelKwargs
        Keyword arguments to pass to the `ArrayDisplayModel` constructor. If
        `display_model` is provided, these will be ignored.
    """

    def __init__(
        self,
        data_or_model: Any | DataWrapper | ArrayDataDisplayModel = None,
        /,
        display_model: ArrayDisplayModel | None = None,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ) -> None:
        if isinstance(data_or_model, ArrayDataDisplayModel):
            if display_model is not None or kwargs:
                warnings.warn(
                    "If an ArrayDataDisplayModel is provided, display_model and kwargs "
                    "will be ignored.",
                    stacklevel=2,
                )
            model = data_or_model
        else:
            if display_model is not None and kwargs:
                warnings.warn(
                    "If display_model is provided, kwargs will be ignored.",
                    stacklevel=2,
                )
            display_model = display_model or ArrayDisplayModel(**kwargs)
            model = ArrayDataDisplayModel(
                data_wrapper=data_or_model, display=display_model
            )

        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[LutKey, ChannelController] = {}

        # get and create the front-end and canvas classes
        frontend_cls = _app.get_array_view_class()
        canvas_cls = _app.get_array_canvas_class()
        self._canvas = canvas_cls()
        self._canvas.set_ndim(2)

        self._histogram: HistogramCanvas | None = None
        self._view = frontend_cls(self._canvas.frontend_widget())

        self._model: ArrayDataDisplayModel = model
        self._set_model_connected(self._model.display)

        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.histogramRequested.connect(self.add_histogram)
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)
        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)

        if self._model.data_wrapper is not None:
            self._fully_synchronize_view()

    # -------------- possibly move this logic up to DataDisplayModel --------------
    @property
    def view(self) -> ArrayView:
        """Return the front-end view object."""
        return self._view

    @property
    def model(self) -> ArrayDataDisplayModel:
        """Return the display model for the viewer."""
        return self._model

    @model.setter
    def model(self, model: ArrayDisplayModel | ArrayDataDisplayModel) -> None:
        """Set the display model for the viewer."""
        self._set_model_connected(self._model.display, False)
        if isinstance(model, ArrayDisplayModel):
            self._model.display = model
        elif isinstance(model, ArrayDataDisplayModel):
            self._model = model
        else:
            raise TypeError(
                "model must be an ArrayDisplayModel or ArrayDataDisplayModel"
            )
        self._set_model_connected(self._model.display)
        self._fully_synchronize_view()

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._model.data_wrapper is None:
            return None  # pragma: no cover
        # returning the actual data, not the wrapper
        return self._model.data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        self._model.data = data
        self._fully_synchronize_view()

    def show(self) -> None:
        """Show the viewer."""
        self._view.set_visible(True)

    def hide(self) -> None:
        """Show the viewer."""
        self._view.set_visible(False)

    def close(self) -> None:
        """Close the viewer."""
        self._view.set_visible(False)

    def add_histogram(self) -> None:
        histogram_cls = _app.get_histogram_canvas_class()  # will raise if not supported
        self._histogram = histogram_cls()
        self._view.add_histogram(self._histogram.frontend_widget())
        for view in self._lut_controllers.values():
            view.add_lut_view(self._histogram)
            # FIXME: hack
            if handles := view.handles:
                data = handles[0].data()
                counts, edges = _calc_hist_bins(data)
                self._histogram.set_data(counts, edges)

        if self.data is not None:
            self._update_hist_domain_for_dtype()

    # --------------------- PRIVATE ------------------------------------------

    def _update_hist_domain_for_dtype(
        self, dtype: np.typing.DTypeLike | None = None
    ) -> None:
        if self._histogram is None:
            return
        if dtype is None:
            if (wrapper := self.model.data_wrapper) is None:
                return
            dtype = wrapper.dtype
        else:
            dtype = np.dtype(dtype)
        if dtype.kind in "iu":
            iinfo = np.iinfo(dtype)
            self._histogram.set_range(x=(iinfo.min, iinfo.max))

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
        display_model = self._model.display
        with self.view.currentIndexChanged.blocked():
            self._view.create_sliders(self._model.normed_data_coords)
        self._view.set_channel_mode(display_model.channel_mode)
        if self.data is not None:
            self._update_visible_sliders()
            self._view.set_current_index(display_model.current_index)
            if wrapper := self._model.data_wrapper:
                self._view.set_data_info(wrapper.summary_info())

            self._clear_canvas()
            self._update_canvas()
            for lut_ctr in self._lut_controllers.values():
                lut_ctr._update_view_from_model()
            self._update_hist_domain_for_dtype()

    def _on_model_visible_axes_changed(self) -> None:
        self._update_visible_sliders()
        self._update_canvas()

    def _on_model_current_index_changed(self) -> None:
        value = self._model.display.current_index
        self._view.set_current_index(value)
        self._update_canvas()

    def _on_model_channel_mode_changed(self, mode: ChannelMode) -> None:
        self._view.set_channel_mode(mode)
        self._update_visible_sliders()
        show_channel_luts = mode in {ChannelMode.COLOR, ChannelMode.COMPOSITE}
        for lut_ctrl in self._lut_controllers.values():
            for view in lut_ctrl.lut_views:
                if lut_ctrl.key is None:
                    view.set_visible(not show_channel_luts)
                else:
                    view.set_visible(show_channel_luts)
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
        self._model.display.current_index.update(self._view.current_index())

    def _on_view_reset_zoom_clicked(self) -> None:
        """Reset the zoom level of the canvas."""
        self._canvas.set_range()

    def _on_canvas_mouse_moved(self, event: MouseMoveEvent) -> None:
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
        self._model.display.channel_mode = mode

    # ------------------ Helper methods ------------------

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        hidden_sliders: tuple[int, ...] = self._model.normed_visible_axes
        if self._model.display.channel_mode.is_multichannel():
            if ch := self._model.normed_channel_axis:
                hidden_sliders += (ch,)

        self._view.hide_sliders(hidden_sliders, show_remainder=True)

    def _update_canvas(self) -> None:
        """Force the canvas to fetch and update the displayed data.

        This is called (frequently) when anything changes that requires a redraw.
        It fetches the current data slice from the model and updates the image handle.
        """
        if not self._model.data_wrapper:
            return  # pragma: no cover

        display_model = self._model.display
        # TODO: make asynchronous
        for future in self._model.request_sliced_data():
            response = future.result()
            key = response.channel_key
            data = response.data

            if (lut_ctrl := self._lut_controllers.get(key)) is None:
                if key is None:
                    model = display_model.default_lut
                elif key in display_model.luts:
                    model = display_model.luts[key]
                else:
                    # we received a new channel key that has not been set in the model
                    # so we create a new LUT model for it
                    model = display_model.luts[key] = LUTModel()

                lut_views = [self._view.add_lut_view()]
                if self._histogram is not None:
                    lut_views.append(self._histogram)
                self._lut_controllers[key] = lut_ctrl = ChannelController(
                    key=key,
                    model=model,
                    views=lut_views,
                )

            if not lut_ctrl.handles:
                # we don't yet have any handles for this channel
                lut_ctrl.add_handle(self._canvas.add_image(data))
            else:
                lut_ctrl.update_texture_data(data)
                if self._histogram is not None:
                    # TODO: once data comes in in chunks, we'll need a proper stateful
                    # stats object that calculates the histogram incrementally
                    counts, bin_edges = _calc_hist_bins(data)
                    # TODO: currently this is updating the histogram on *any*
                    # channel index... so it doesn't work with composite mode
                    self._histogram.set_data(counts, bin_edges)
                    self._histogram.set_range()

        self._canvas.refresh()

    def _get_values_at_world_point(self, x: int, y: int) -> dict[LutKey, float]:
        # TODO: handle 3D data
        if (
            x < 0 or y < 0
        ) or self._model.display.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[LutKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((y, x))) is not None:
                values[key] = value

        return values


def _calc_hist_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxval = np.iinfo(data.dtype).max
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    return counts, bin_edges
