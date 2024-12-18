from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controller._channel_controller import ChannelController
from ndv.models import DataDisplayModel, RectangularROIModel
from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import LUTModel
from ndv.models._viewer_model import CanvasMode, ViewerModel
from ndv.views import (
    get_canvas_class,
    get_histogram_canvas_class,
    get_view_frontend_class,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from ndv._types import MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModel
    from ndv.views.bases import ArrayView, HistogramCanvas
    from ndv.views.bases.graphics._canvas_elements import BoundingBox

    LutKey: TypeAlias = int | None


# (probably rename to something like just Viewer...
# or compose all three of the model/view/controller into a single class
# that exposes the public interface)


class ViewerController:
    """The controller mostly manages the connection between the model and the view.

    This is the primary public interface for the viewer.
    """

    def __init__(
        self,
        data: DataDisplayModel | None = None,
        roi: RectangularROIModel | None = None,
        viewer: ViewerModel | None = None,
    ) -> None:
        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[LutKey, ChannelController] = {}

        self._view_model = viewer or ViewerModel()

        # get and create the front-end and canvas classes
        frontend_cls = get_view_frontend_class()
        canvas_cls = get_canvas_class()
        self._canvas = canvas_cls(self._view_model)
        self._canvas.set_ndim(2)

        self._histogram: HistogramCanvas | None = None
        self._view = frontend_cls(self._canvas, self._view_model)
        # TODO: _dd_model is perhaps a temporary concept, and definitely name
        self._dd_model = data or DataDisplayModel()
        # FIXME
        self._roi = roi or RectangularROIModel()
        self._bb: BoundingBox = self._canvas.add_bounding_box()

        self._set_model_connected(self._dd_model.display)
        self._set_roi_connected(self._roi)

        self._view_model.events.mode.connect(self._on_view_model_mode_changed)

        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.histogramRequested.connect(self.add_histogram)
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)

        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)

        self._bb.boundingBoxChanged.connect(self._on_view_bounding_box_changed)

    # -------------- possibly move this logic up to DataDisplayModel --------------
    @property
    def view(self) -> ArrayView:
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
        self._update_hist_domain_for_dtype(data.dtype)

    @property
    def roi(self) -> RectangularROIModel:
        return self._roi

    # -----------------------------------------------------------------------------

    def add_histogram(self) -> None:
        histogram_cls = get_histogram_canvas_class()  # will raise if not supported
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
            self._update_hist_domain_for_dtype(self.data.dtype)

    def _update_hist_domain_for_dtype(self, dtype: np.typing.DTypeLike) -> None:
        if self._histogram is None:
            return

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

    # FIXME
    def _set_roi_connected(
        self, model: RectangularROIModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.  (but it's kinda ugly)
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.bounding_box, self._on_roi_bounding_box_changed),
            (model.events.visible, self._on_roi_visibility_changed),
        ]:
            getattr(obj, _connect)(callback)
            if _connect == "connect":
                callback()

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

    def _on_roi_bounding_box_changed(self) -> None:
        box_min, box_max = self._roi.bounding_box
        self._bb.set_bounding_box(box_min, box_max)

    def _on_roi_visibility_changed(self) -> None:
        self._bb.set_visible(self._roi.visible)

    def _on_view_model_mode_changed(self, mode: CanvasMode) -> None:
        if mode == CanvasMode.CREATE_ROI:
            # Discard the old ROI
            # TODO: Support multiple ROIS
            self._set_roi_connected(self._roi, connect=False)
            self._bb.remove()
            # Create a new ROI
            self._bb = self._canvas.add_bounding_box()
            self._bb.boundingBoxChanged.connect(self._on_view_bounding_box_changed)
            self._set_roi_connected(self._roi, connect=True)

    # ------------------ View callbacks ------------------

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self.model.current_index.update(self._view.current_index())

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
        self.model.channel_mode = mode

    def _on_view_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        self._roi.bounding_box = bb

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
        if self._dd_model.data_wrapper:
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
        if (x < 0 or y < 0) or self.model.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[LutKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((y, x))) is not None:
                values[key] = value

        return values

    def show(self) -> None:
        """Show the viewer."""
        self._view.set_visible(True)


def _calc_hist_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxval = np.iinfo(data.dtype).max
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    return counts, bin_edges
