from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import ArrayDisplayModel, ChannelMode, DataWrapper, LUTModel
from ndv.models._roi_model import RectangularROIModel
from ndv.models._viewer_model import InteractionMode
from ndv.views import _app

from ._base_array_viewer import _BaseArrayViewer

if TYPE_CHECKING:
    from collections.abc import Hashable
    from concurrent.futures import Future
    from typing import Any, Unpack

    from ndv._types import ChannelKey
    from ndv.models._array_display_model import ArrayDisplayModelKwargs
    from ndv.models._data_display_model import DataResponse
    from ndv.views.bases import HistogramCanvas
    from ndv.views.bases._graphics._canvas_elements import RectangularROIHandle


class ArrayViewer(_BaseArrayViewer):
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
        super().__init__(data, display_model=display_model, **kwargs)

        self._viewer_model.events.interaction_mode.connect(
            self._on_interaction_mode_changed
        )
        self._roi_model: RectangularROIModel | None = None

        app = _app.gui_frontend()
        # whether to fetch data asynchronously.  Not publicly exposed yet...
        # but can use 'NDV_SYNCHRONOUS' env var to set globally
        # jupyter doesn't need async because it's already async (in that the
        # GUI is already running in JS)
        NDV_SYNCHRONOUS = os.getenv("NDV_SYNCHRONOUS", "0") in {"1", "True", "true"}
        self._async = not NDV_SYNCHRONOUS and app != _app.GuiFrontend.JUPYTER
        # set of futures for data requests
        self._futures: set[Future[DataResponse]] = set()
        self._histograms: dict[ChannelKey, HistogramCanvas] = {}
        self._roi_view: RectangularROIHandle | None = None

        self._set_model_connected(self._data_model.display)

        self._view.histogramRequested.connect(self._add_histogram)

        if self._data_model.data_wrapper is not None:
            self._fully_synchronize_view()

    # -------------- public attributes and methods -------------------------

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

    @property
    def roi(self) -> RectangularROIModel | None:
        """Return ROI being displayed."""
        return self._roi_model

    @roi.setter
    def roi(self, roi_model: RectangularROIModel | None) -> None:
        """Set ROI being displayed."""
        # Disconnect old model
        if self._roi_model is not None:
            self._set_roi_model_connected(self._roi_model, False)

        # Connect new model
        if isinstance(roi_model, tuple):
            self._roi_model = RectangularROIModel(bounding_box=roi_model)
        else:
            self._roi_model = roi_model
        if self._roi_model is not None:
            self._set_roi_model_connected(self._roi_model)
        self._synchronize_roi()

    # --------------------- PRIVATE ------------------------------------------

    def _add_histogram(self, channel: ChannelKey = None) -> None:
        histogram_cls = _app.get_histogram_canvas_class()  # will raise if not supported
        hist = histogram_cls()
        if ctrl := self._lut_controllers.get(channel, None):
            self._view.add_histogram(channel, hist.frontend_widget())
            ctrl.add_lut_view(hist)
            # FIXME: hack
            if handles := ctrl.handles:
                data = handles[0].data()
                counts, edges = _calc_hist_bins(data)
                hist.set_data(counts, edges)

        self._histograms[channel] = hist
        if self.data is not None:
            self._update_hist_domain_for_dtype()

    def _update_hist_domain_for_dtype(
        self, dtype: np.typing.DTypeLike | None = None
    ) -> None:
        if len(self._histograms) == 0:
            return
        if dtype is None:
            if (wrapper := self._data_model.data_wrapper) is None:
                return
            dtype = wrapper.dtype
        else:
            dtype = np.dtype(dtype)
        if dtype.kind in "iu":
            iinfo = np.iinfo(dtype)
            for hist in self._histograms.values():
                hist.set_range(x=(iinfo.min, iinfo.max))

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

    def _set_roi_model_connected(
        self, model: RectangularROIModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.  (but it's kinda ugly)
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.bounding_box, self._on_roi_model_bounding_box_changed),
            (model.events.visible, self._on_roi_model_visible_changed),
        ]:
            getattr(obj, _connect)(callback)

        if _connect:
            self._create_roi_view()
        else:
            if self._roi_view:
                self._roi_view.remove()

    # ------------------ Model callbacks ------------------

    def _fully_synchronize_view(self) -> None:
        """Fully re-synchronize the view with the model."""
        display_model = self._data_model.display
        self._view.set_channel_mode(display_model.channel_mode)
        if (wrapper := self._data_model.data_wrapper) is not None:
            with self._view.currentIndexChanged.blocked():
                self._view.create_sliders(wrapper.coords)
            self._view.set_visible_axes(self._data_model.normed_visible_axes)
            self._update_visible_sliders()
            if cur_index := display_model.current_index:
                self._view.set_current_index(cur_index)
            # reconcile view sliders with model
            self._on_view_current_index_changed()
            self._view.set_data_info(wrapper.summary_info())

            self._clear_canvas()
            self._request_data()
            for lut_ctr in self._lut_controllers.values():
                lut_ctr.synchronize()
            self._update_hist_domain_for_dtype()
        self._synchronize_roi()

    def _synchronize_roi(self) -> None:
        """Fully re-synchronize the ROI view with the model."""
        if self.roi is not None:
            self._on_roi_model_bounding_box_changed(self.roi.bounding_box)
            self._on_roi_model_visible_changed(self.roi.visible)

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

    def _on_roi_model_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        if self._roi_view is not None:
            self._roi_view.set_bounding_box(*bb)

    def _on_roi_model_visible_changed(self, visible: bool) -> None:
        if self._roi_view is not None:
            self._roi_view.set_visible(visible)

    def _on_interaction_mode_changed(self, mode: InteractionMode) -> None:
        if mode == InteractionMode.CREATE_ROI:
            # Create ROI model if needed to store ROI state
            if self.roi is None:
                self.roi = RectangularROIModel(visible=False)

            # Create a new ROI
            self._create_roi_view()

    def _create_roi_view(self) -> None:
        # Remove old ROI view
        # TODO: Enable multiple ROIs
        if self._roi_view:
            self._roi_view.remove()

        # Create new ROI view
        self._roi_view = self._canvas.add_bounding_box()
        # Connect view signals
        self._roi_view.boundingBoxChanged.connect(
            self._on_roi_view_bounding_box_changed
        )

    def _clear_canvas(self) -> None:
        for lut_ctrl in self._lut_controllers.values():
            # self._view.remove_lut_view(lut_ctrl.lut_view)
            while lut_ctrl.handles:
                lut_ctrl.handles.pop().remove()
        # do we need to cleanup the lut views themselves?

    # ------------------ View callbacks ------------------

    def _on_roi_view_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        if self._roi_model:
            self._roi_model.bounding_box = bb

    # ------------------ Helper methods ------------------

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        if self._data_model.data_wrapper is None:
            return
        hidden_indices: set[int] = set(self._data_model.normed_visible_axes)
        if self._data_model.display.channel_mode.is_multichannel():
            if (ch := self._data_model.normed_channel_axis) is not None:
                hidden_indices.add(ch)

        # hide singleton axes
        for ax, coord in self._data_model.normed_data_coords.items():
            if len(coord) < 2:
                hidden_indices.add(ax)

        # here we look into the *non*-normalized wrapper.dims names
        # and add those to the hidden indices as well (so that sliders are hidden
        # regardless of the form in which they were expressed in the model)
        hidden_sliders: set[Hashable] = set(hidden_indices)
        if wrapper := self._data_model.data_wrapper:
            for hidden in list(hidden_indices):
                hidden_sliders.add(wrapper.dims[hidden])

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
            self._viewer_model.show_progress_spinner = True

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
        self._viewer_model.show_progress_spinner = False

    @_app.ensure_main_thread
    def _on_data_response_ready(self, future: Future[DataResponse]) -> None:
        # NOTE: removing the reference to the last future here is important
        # because the future has a reference to this widget in its _done_callbacks
        # which will prevent the widget from being garbage collected if the future
        self._futures.discard(future)
        if not self._futures:
            self._viewer_model.show_progress_spinner = False

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

                lut_views = [self._view.add_lut_view(key)]
                if hist := self._histograms.get(key, None):
                    lut_views.append(hist)
                self._lut_controllers[key] = lut_ctrl = ChannelController(
                    key=key,
                    lut_model=model,
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

            if hist := self._histograms.get(key, None):
                # TODO: once data comes in in chunks, we'll need a proper stateful
                # stats object that calculates the histogram incrementally
                counts, bin_edges = _calc_hist_bins(data)
                # FIXME: currently this is updating the histogram on *any*
                # channel index... so it doesn't work with composite mode
                hist.set_data(counts, bin_edges)
                hist.set_range()

        self._canvas.refresh()


def _calc_hist_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxval = np.iinfo(data.dtype).max
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    return counts, bin_edges
