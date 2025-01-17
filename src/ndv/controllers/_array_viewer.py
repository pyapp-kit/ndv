from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import ArrayDisplayModel, ChannelMode, DataWrapper, LUTModel
from ndv.models._data_display_model import DataResponse, _ArrayDataDisplayModel
from ndv.views import _app

if TYPE_CHECKING:
    from concurrent.futures import Future
    from typing import Any, Unpack

    from typing_extensions import TypeAlias

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModelKwargs
    from ndv.views.bases import HistogramCanvas

    LutKey: TypeAlias = int | None


class ArrayViewer:
    """Viewer dedicated to displaying a single n-dimensional array.

    This wraps a model and sview into a single object, and defines the
    public API.

    !!! tip "See also"

        [**`ndv.imshow`**][ndv.imshow] - a convenience function that constructs and
        shows an `ArrayViewer`.

    !!! note "Future plans"

        In the future, `ndv` would like to support multiple, layered data sources with
        coordinate transforms. We reserve the name `Viewer` for a more fully featured
        viewer. `ArrayViewer` assumes you're viewing a single array.

    Parameters
    ----------
    data :  DataWrapper | Any
        Data to be displayed.
    display_model : ArrayDisplayModel, optional
        Just the display model to use. If provided, `data_or_model` must be an array
        or `DataWrapper`... and kwargs will be ignored.
    **kwargs: ArrayDisplayModelKwargs
        Keyword arguments to pass to the `ArrayDisplayModel` constructor. If
        `display_model` is provided, these will be ignored.
    """

    def __init__(
        self,
        data: Any | DataWrapper = None,
        /,
        display_model: ArrayDisplayModel | None = None,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ) -> None:
        if display_model is not None and kwargs:
            warnings.warn(
                "When display_model is provided, kwargs are be ignored.",
                stacklevel=2,
            )
        self._data_model = _ArrayDataDisplayModel(
            data_wrapper=data, display=display_model or ArrayDisplayModel(**kwargs)
        )

        app = _app.gui_frontend()

        # whether to fetch data asynchronously.  Not publicly exposed yet...
        # but can use 'NDV_SYNCHRONOUS' env var to set globally
        # jupyter doesn't need async because it's already async (in that the
        # GUI is already running in JS)
        NDV_SYNCHRONOUS = os.getenv("NDV_SYNCHRONOUS", "0") in {"1", "True", "true"}
        self._async = not NDV_SYNCHRONOUS and app != _app.GuiFrontend.JUPYTER
        # set of futures for data requests
        self._futures: set[Future[DataResponse]] = set()

        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[LutKey, ChannelController] = {}

        # get and create the front-end and canvas classes
        frontend_cls = _app.get_array_view_class()
        canvas_cls = _app.get_array_canvas_class()
        self._canvas = canvas_cls()

        self._histogram: HistogramCanvas | None = None
        self._view = frontend_cls(self._canvas.frontend_widget(), self._data_model)

        self._set_model_connected(self._data_model.display)
        self._canvas.set_ndim(self.display_model.n_visible_axes)

        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.histogramRequested.connect(self._add_histogram)
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)
        self._view.visibleAxesChanged.connect(self._on_view_visible_axes_changed)

        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)

        if self._data_model.data_wrapper is not None:
            self._fully_synchronize_view()

    # -------------- public attributes and methods -------------------------

    # @property
    # def view(self) -> ArrayView:
    #     return self._view

    def widget(self) -> Any:
        """Return the native front-end widget.

        !!! Warning

            If you directly manipulate the frontend widget, you're on your own :smile:.
            No guarantees can be made about synchronization with the model.  It is
            exposed for embedding in an application, and for experimentation and custom
            use cases.  Please [open an
            issue](https://github.com/pyapp-kit/ndv/issues/new) if you have questions.
        """
        return self._view.frontend_widget()

    @property
    def display_model(self) -> ArrayDisplayModel:
        """Return the current ArrayDisplayModel."""
        return self._data_model.display

    @display_model.setter
    def display_model(self, model: ArrayDisplayModel) -> None:
        """Set the ArrayDisplayModel."""
        if not isinstance(model, ArrayDisplayModel):  # pragma: no cover
            raise TypeError("model must be an ArrayDisplayModel")
        self._set_model_connected(self._data_model.display, False)
        self._data_model.display = model
        self._set_model_connected(self._data_model.display)
        self._fully_synchronize_view()

    @property
    def data_wrapper(self) -> Any:
        """Return data being displayed."""
        return self._data_model.data_wrapper

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._data_model.data_wrapper is None:
            return None  # pragma: no cover
        # returning the actual data, not the wrapper
        return self._data_model.data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        if data is None:
            self._data_model.data_wrapper = None
        else:
            self._data_model.data_wrapper = DataWrapper.create(data)
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

    def clone(self) -> ArrayViewer:
        """Return a new ArrayViewer instance with the same data and display model.

        Currently, this is a shallow copy.  Modifying one viewer will affect the state
        of the other.
        """
        # TODO: provide deep_copy option
        return ArrayViewer(
            self._data_model.data_wrapper, display_model=self.display_model
        )

    # --------------------- PRIVATE ------------------------------------------

    def _add_histogram(self) -> None:
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

    def _update_hist_domain_for_dtype(
        self, dtype: np.typing.DTypeLike | None = None
    ) -> None:
        if self._histogram is None:
            return
        if dtype is None:
            if (wrapper := self._data_model.data_wrapper) is None:
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
        display_model = self._data_model.display
        with self._view.currentIndexChanged.blocked():
            self._view.create_sliders(self._data_model.normed_data_coords)
        self._view.set_channel_mode(display_model.channel_mode)
        if self.data is not None:
            self._view.set_visible_axes(self._data_model.normed_visible_axes)
            self._update_visible_sliders()
            if cur_index := display_model.current_index:
                self._view.set_current_index(cur_index)
            # reconcile view sliders with model
            self._on_view_current_index_changed()
            if wrapper := self._data_model.data_wrapper:
                self._view.set_data_info(wrapper.summary_info())

            self._clear_canvas()
            self._request_data()
            for lut_ctr in self._lut_controllers.values():
                lut_ctr._update_view_from_model()
            self._update_hist_domain_for_dtype()

    def _on_model_visible_axes_changed(self) -> None:
        self._view.set_visible_axes(self._data_model.normed_visible_axes)
        self._update_visible_sliders()
        self._clear_canvas()
        self._canvas.set_ndim(self.display_model.n_visible_axes)
        self._request_data()

    def _on_model_current_index_changed(self) -> None:
        value = self._data_model.display.current_index
        self._view.set_current_index(value)
        self._request_data()

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
        self._request_data()

    def _clear_canvas(self) -> None:
        for lut_ctrl in self._lut_controllers.values():
            # self._view.remove_lut_view(lut_ctrl.lut_view)
            while lut_ctrl.handles:
                lut_ctrl.handles.pop().remove()
        # do we need to cleanup the lut views themselves?

    # ------------------ View callbacks ------------------

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self._data_model.display.current_index.update(self._view.current_index())

    def _on_view_visible_axes_changed(self) -> None:
        """Update the model when the visible axes change."""
        self.display_model.visible_axes = self._view.visible_axes()  # type: ignore [assignment]

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
        self._data_model.display.channel_mode = mode

    # ------------------ Helper methods ------------------

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        hidden_sliders: tuple[int, ...] = self._data_model.normed_visible_axes
        if self._data_model.display.channel_mode.is_multichannel():
            if ch := self._data_model.normed_channel_axis:
                hidden_sliders += (ch,)

        self._view.hide_sliders(hidden_sliders, show_remainder=True)

    # The request cycle looks like this:
    # 1. something changes on the model requiring new data
    # 2. _request_data is called
    # 3. _data_model.request_sliced_data returns a list of futures
    # 4. each future is connected to `_on_data_response_ready` and stored.
    # 5. when the future resolves, `_on_data_response_ready` draws the response.

    def _request_data(self) -> None:
        """Fetch and update the displayed data.

        This is called (frequently) when anything changes that requires a redraw.
        It fetches the current data slice from the model and updates the image handle.
        """
        if not self._data_model.data_wrapper:
            return  # pragma: no cover

        self._cancel_futures()
        for future in self._data_model.request_sliced_data(self._async):
            self._futures.add(future)
            future.add_done_callback(self._on_data_response_ready)

        if self._futures:
            self._view.set_progress_spinner_visible(True)

    def _is_idle(self) -> bool:
        """Return True if no futures are running. Used for testing, and debugging."""
        return all(f.done() for f in self._futures)

    def _join(self) -> None:
        """Block until all futures are done. Used for testing, and debugging."""
        for future in self._futures:
            future.result()

    def _cancel_futures(self) -> None:
        while self._futures:
            self._futures.pop().cancel()
        self._futures.clear()
        self._view.set_progress_spinner_visible(False)

    @_app.ensure_main_thread
    def _on_data_response_ready(self, future: Future[DataResponse]) -> None:
        # NOTE: removing the reference to the last future here is important
        # because the future has a reference to this widget in its _done_callbacks
        # which will prevent the widget from being garbage collected if the future
        self._futures.discard(future)
        if not self._futures:
            self._view.set_progress_spinner_visible(False)

        if future.cancelled():
            return

        try:
            response = future.result()
        except Exception as e:
            warnings.warn(f"Error fetching data: {e}", stacklevel=2)
            return

        display_model = self._data_model.display
        for key, data in response.data.items():
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
                if response.n_visible_axes == 2:
                    handle = self._canvas.add_image(data)
                    lut_ctrl.add_handle(handle)
                elif response.n_visible_axes == 3:
                    handle = self._canvas.add_volume(data)
                    lut_ctrl.add_handle(handle)

            else:
                lut_ctrl.update_texture_data(data)

            if self._histogram is not None:
                # TODO: once data comes in in chunks, we'll need a proper stateful
                # stats object that calculates the histogram incrementally
                counts, bin_edges = _calc_hist_bins(data)
                # FIXME: currently this is updating the histogram on *any*
                # channel index... so it doesn't work with composite mode
                self._histogram.set_data(counts, bin_edges)
                self._histogram.set_range()

        self._canvas.refresh()

    def _get_values_at_world_point(self, x: int, y: int) -> dict[LutKey, float]:
        # TODO: handle 3D data
        if (
            x < 0 or y < 0
        ) or self._data_model.display.n_visible_axes != 2:  # pragma: no cover
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
