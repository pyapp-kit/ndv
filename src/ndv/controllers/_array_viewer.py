from __future__ import annotations

import os
import warnings
from concurrent.futures import Future
from contextlib import suppress
from itertools import count
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from psygnal import Signal

from ndv._keybindings import handle_key_press
from ndv.controllers._channel_controller import ChannelController
from ndv.controllers._image_stats import ImageStats, compute_image_stats
from ndv.models import ArrayDisplayModel, ChannelMode, DataWrapper, LUTModel
from ndv.models._lut_model import ClimsManual
from ndv.models._resolve import (
    EMPTY_STATE,
    DataResponse,
    ResolvedDisplayState,
    build_slice_requests,
    process_request,
    resolve,
)
from ndv.models._roi_model import RectangularROIModel
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views import _app

if TYPE_CHECKING:
    from typing import Any

    import cmap as cmap_mod
    import numpy.typing as npt
    from typing_extensions import Unpack

    from ndv._types import AxisKey, ChannelKey, KeyPressEvent, MouseMoveEvent
    from ndv.models._array_display_model import ArrayDisplayModelKwargs
    from ndv.models._viewer_model import ArrayViewerModelKwargs
    from ndv.views.bases import HistogramCanvas, SharedHistogramCanvas
    from ndv.views.bases._graphics._canvas_elements import RectangularROIHandle


class ArrayViewer:
    """Viewer dedicated to displaying a single n-dimensional array.

    This wraps a model and view into a single object, and defines the
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

    stats_updated = Signal(object, ImageStats)

    def __init__(
        self,
        data: Any | DataWrapper = None,
        /,
        *,
        viewer_options: ArrayViewerModel | ArrayViewerModelKwargs | None = None,
        display_model: ArrayDisplayModel | None = None,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ) -> None:
        wrapper = None if data is None else DataWrapper.create(data)
        if display_model is None:
            display_model = self._default_display_model(wrapper, **kwargs)
        elif kwargs:
            warnings.warn(
                "When display_model is provided, kwargs are be ignored.",
                stacklevel=2,
            )

        self._display_model = display_model
        self._data_wrapper: DataWrapper | None = wrapper
        self._resolved = EMPTY_STATE

        self._connect_datawrapper(None, wrapper)

        self._viewer_model = ArrayViewerModel.model_validate(viewer_options or {})
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
        self._async = not NDV_SYNCHRONOUS and app not in (
            _app.GuiFrontend.JUPYTER,
            _app.GuiFrontend.MARIMO,
        )
        # maps pending futures to their request generation (for stale detection)
        self._gen_counter = count()
        self._current_gen: int = 0
        self._futures: dict[Future[DataResponse], int] = {}

        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[ChannelKey, ChannelController] = {}

        # get and create the front-end and canvas classes
        frontend_cls = _app.get_array_view_class()
        canvas_cls = _app.get_array_canvas_class()
        self._canvas = canvas_cls(self._viewer_model)

        # TODO: Is this necessary?
        self._histograms: dict[ChannelKey, HistogramCanvas] = {}
        self._shared_histogram: SharedHistogramCanvas | None = None
        self._shared_histogram_links: dict[ChannelKey, _SharedHistogramLink] = {}

        # Pre-create the shared histogram canvas now (on the main thread).
        # On macOS + jupyter_rfb backend, vispy creates a hidden GLFW window
        # per canvas for the GL context, and GLFW/NSWindow creation must happen
        # on the main thread. The histogram button handler runs on the kernel's
        # I/O thread, so we must create the canvas here to avoid a crash.
        self._precreated_shared_histogram: SharedHistogramCanvas | None = None
        if app in (_app.GuiFrontend.JUPYTER, _app.GuiFrontend.MARIMO):
            hist_cls = _app.get_shared_histogram_canvas_class()
            self._precreated_shared_histogram = hist_cls()

        self._view = frontend_cls(self._canvas.frontend_widget(), self._viewer_model)
        if self._precreated_shared_histogram is not None:
            self._view.set_histogram_widget(
                self._precreated_shared_histogram.frontend_widget()
            )

        self._roi_view: RectangularROIHandle | None = None

        self._set_model_connected(self._display_model)
        self._canvas.set_ndim(self._display_model.n_visible_axes)

        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.histogramRequested.connect(self._add_histogram)
        self._view.sharedHistogramRequested.connect(self._add_shared_histogram)
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)
        self._view.ndimToggleRequested.connect(self._on_view_ndim_toggle_requested)

        self._highlight_pos: tuple[float, float] | None = None
        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)
        self._canvas.mouseLeft.connect(self._on_canvas_mouse_left)

        self._focused_slider_axis: AxisKey | None = None
        self._disconnect_key_events = _app.filter_key_events(
            self._view.frontend_widget(), self._view
        )
        self._view.keyPressed.connect(self._on_key_pressed)

        if self._data_wrapper is not None:
            self._fully_synchronize_view()

    # -------------- public attributes and methods -------------------------

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
        return self._display_model

    @display_model.setter
    def display_model(self, model: ArrayDisplayModel) -> None:
        """Set the ArrayDisplayModel."""
        if not isinstance(model, ArrayDisplayModel):  # pragma: no cover
            raise TypeError("model must be an ArrayDisplayModel")
        self._set_model_connected(self._display_model, False)
        self._display_model = model
        self._set_model_connected(self._display_model)
        self._fully_synchronize_view()

    @property
    def data_wrapper(self) -> Any:
        """Return the data wrapper object being used to interface with the data."""
        return self._data_wrapper

    @property
    def data(self) -> Any:
        """Return data being displayed (the actual data, not the wrapper)."""
        if self._data_wrapper is None:
            return None  # pragma: no cover
        # returning the actual data, not the wrapper
        return self._data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        _new = None if data is None else DataWrapper.create(data)
        old = self._data_wrapper
        self._data_wrapper = _new
        self._connect_datawrapper(old, _new)
        self._fully_synchronize_view()

    @property
    def roi(self) -> RectangularROIModel | None:
        """Return ROI being displayed."""
        return self._roi_model

    @roi.setter
    def roi(self, roi_model: RectangularROIModel | tuple | None) -> None:
        """Set ROI being displayed.

        Either a RectangularROIModel or a tuple of ((x1, y1), (x2, y2)) can be provided.
        Bounding box is in data coordinates (i.e. array indices).
        """
        # Disconnect old model
        if self._roi_model is not None:
            self._set_roi_model_connected(self._roi_model, False)

        if roi_model is None:
            self._roi_model = None
        else:
            self._roi_model = RectangularROIModel.model_validate(roi_model)
            self._set_roi_model_connected(self._roi_model)
        self._synchronize_roi()

    def show(self) -> None:
        """Show the viewer."""
        self._view.set_visible(True)

    def hide(self) -> None:
        """Hide the viewer."""
        self._view.set_visible(False)

    def close(self) -> None:
        """Close the viewer."""
        self._disconnect_key_events()
        self._view.set_visible(False)

    def clone(self) -> ArrayViewer:
        """Return a new ArrayViewer instance with the same data and display model.

        Currently, this is a shallow copy.  Modifying one viewer will affect the state
        of the other.
        """
        # TODO: provide deep_copy option
        return ArrayViewer(self._data_wrapper, display_model=self.display_model)

    def refresh_stats(self) -> None:
        """Force re-emit stats for all channels with existing data.

        This will mostly be used by external listeners that want the initial data,
        before any interaction has occurred.
        """
        if not len(self.stats_updated):
            return
        sig_bits = wrp.significant_bits if (wrp := self._data_wrapper) else None
        for key, ctrl in self._lut_controllers.items():
            if ctrl.handles:
                stats = compute_image_stats(
                    ctrl.handles[0].data(),
                    ctrl.lut_model.clims,
                    need_histogram=True,
                    significant_bits=sig_bits,
                )
                self.stats_updated.emit(key, stats)

    # --------------------- PRIVATE ------------------------------------------

    def _connect_datawrapper(
        self, old: DataWrapper | None, new: DataWrapper | None
    ) -> None:
        """Set new datawrapper and hook up events."""
        if old is not None:
            with suppress(Exception):
                old.data_changed.disconnect(self._request_data)
                old.dims_changed.disconnect(self._on_dims_changed)
        if new is not None:
            new.data_changed.connect(self._request_data)
            new.dims_changed.connect(self._on_dims_changed)

    @staticmethod
    def _default_display_model(
        data: None | DataWrapper, **kwargs: Unpack[ArrayDisplayModelKwargs]
    ) -> ArrayDisplayModel:
        """
        Creates a default ArrayDisplayModel when none is provided by the user.

        All magical setup goes here.
        """
        # Can't do any magic with no data
        if data is None:
            return ArrayDisplayModel(**kwargs)

        # cast 3d+ images with shape[-1] of {3,4} to RGB images
        if "channel_mode" not in kwargs and "channel_axis" not in kwargs:
            shape = tuple(data.sizes().values())
            if len(shape) >= 3 and shape[-1] in {3, 4}:
                kwargs["channel_axis"] = -1
                kwargs["channel_mode"] = "rgba"
        return ArrayDisplayModel(**kwargs)

    def _add_histogram(self, channel: ChannelKey = None) -> None:
        histogram_cls = _app.get_histogram_canvas_class()  # will raise if not supported
        hist = histogram_cls()
        self._histograms[channel] = hist

        if ctrl := self._lut_controllers.get(channel, None):
            self._view.add_histogram(channel, hist)
            ctrl.add_lut_view(hist)
            self._connect_histogram(ctrl, hist)

    def _connect_histogram(
        self, ctrl: ChannelController, hist: HistogramCanvas
    ) -> None:
        """Connect a histogram to a channel controller's stats signal."""

        def _on_stats(stats: ImageStats) -> None:
            if stats.counts is not None and stats.bin_edges is not None:
                hist.set_data(stats.counts, stats.bin_edges)

        ctrl.stats_updated.connect(_on_stats)
        # Trigger initial data from existing handle
        if handles := ctrl.handles:
            sig_bits = wrp.significant_bits if (wrp := self._data_wrapper) else None
            ctrl.update_texture_data(handles[0].data(), significant_bits=sig_bits)
        hist.set_range()

    def _add_shared_histogram(self) -> None:
        """Create and connect the shared multi-channel histogram."""
        if self._shared_histogram is not None:
            return
        if self._precreated_shared_histogram is not None:
            hist = self._precreated_shared_histogram
            self._precreated_shared_histogram = None
        else:
            hist_cls = _app.get_shared_histogram_canvas_class()
            hist = hist_cls()
        self._shared_histogram = hist
        self._view.add_shared_histogram(hist)

        # Connect clim/gamma changes from shared histogram back to models
        hist.climsChanged.connect(self._on_shared_histogram_clims_changed)
        hist.gammaChanged.connect(self._on_shared_histogram_gamma_changed)

        # Connect all existing channels
        for key, ctrl in self._lut_controllers.items():
            self._connect_shared_histogram_channel(key, ctrl)

        # Apply current channel mode visibility
        self._update_lut_visibility(self._resolved.channel_mode)
        hist.set_range()

    def _connect_shared_histogram_channel(
        self, key: ChannelKey, ctrl: ChannelController
    ) -> None:
        """Connect a channel controller to the shared histogram."""
        hist = self._shared_histogram
        if hist is None or key in self._shared_histogram_links:
            return

        sig_bits = wrp.significant_bits if (wrp := self._data_wrapper) else None
        self._shared_histogram_links[key] = _SharedHistogramLink(
            key,
            ctrl,
            hist,
            fallback_name=self._fallback_channel_name(key),
            significant_bits=sig_bits,
        )

    def _on_shared_histogram_clims_changed(
        self, key: ChannelKey, clims: tuple[float, float]
    ) -> None:
        if ctrl := self._lut_controllers.get(key):
            ctrl.lut_model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_shared_histogram_gamma_changed(self, key: ChannelKey, gamma: float) -> None:
        if ctrl := self._lut_controllers.get(key):
            ctrl.lut_model.gamma = gamma

    def _update_channel_dtype(
        self, channel: ChannelKey, dtype: npt.DTypeLike | None = None
    ) -> None:
        if not (ctrl := self._lut_controllers.get(channel, None)):
            return
        if dtype is None:
            if self._data_wrapper is None:
                return
            dtype = self._data_wrapper.dtype
        else:
            dtype = np.dtype(dtype)
        if dtype.kind in "iu":
            iinfo = np.iinfo(dtype)
            ctrl.lut_model.clim_bounds = (iinfo.min, iinfo.max)

    def _set_model_connected(
        self, model: ArrayDisplayModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.  (but it's kinda ugly)
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.visible_axes, self._re_resolve),
            (model.events.channel_axis, self._re_resolve),
            (model.current_index.value_changed, self._re_resolve),
            (model.events.channel_mode, self._re_resolve),
            (model.scales.value_changed, self._re_resolve),
            (model.luts.value_changed, self._re_resolve),
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

    # ------------------ Resolve / Apply ------------------

    def _re_resolve(self) -> None:
        """Resolve display state and apply changes."""
        if self._data_wrapper is None:
            return
        old = self._resolved
        resolved = resolve(self._display_model, self._data_wrapper)
        if not self._prepare_channel_mode(resolved):
            return

        self._resolved = resolved
        self._apply_changes(old, resolved)

    def _apply_changes(
        self, old: ResolvedDisplayState, new: ResolvedDisplayState
    ) -> None:
        """Apply diff between old and new resolved state to the view."""
        with self._view.currentIndexChanged.blocked():
            if old.channel_mode != new.channel_mode:
                self._view.set_channel_mode(new.channel_mode)
                self._update_lut_visibility(new.channel_mode)
            if old.visible_axes != new.visible_axes:
                self._view.set_visible_axes(new.visible_axes)
                ndim = len(new.visible_axes)
                self._canvas.set_ndim(cast("Literal[2, 3]", ndim))
                self._clear_canvas()
            if old.hidden_sliders != new.hidden_sliders:
                self._view.hide_sliders(new.hidden_sliders, show_remainder=True)
            if old.current_index != new.current_index:
                self._view.set_current_index(self._display_model.current_index)

        needs_data = (
            old.visible_axes != new.visible_axes
            or old.channel_axis != new.channel_axis
            or old.channel_mode != new.channel_mode
            or old.current_index != new.current_index
        )
        if needs_data:
            self._request_data()

        if old.visible_scales != new.visible_scales:
            self._canvas.set_scales(new.visible_scales)
            self._synchronize_roi()

        if old.channel_axis != new.channel_axis:
            self._push_fallback_channel_names()

        if old.summary_info != new.summary_info:
            self._view.set_data_info(new.summary_info)

    def _fallback_channel_name(self, key: ChannelKey) -> str:
        """Compute the data-derived fallback name for a channel key."""
        if self._data_wrapper is not None and isinstance(key, int):
            names = self._data_wrapper.channel_names(self._resolved.channel_axis)
            return names.get(key, str(key))
        if key is None:
            return ""
        return str(key)

    def _push_fallback_channel_names(self) -> None:
        """Push data-derived fallback names to all LUT views.

        This logic lives here, as opposed to the LUTView or ChannelController, because
        it's one of the few fields that can be resolved from either the model or the
        data, and this is the layer that has access to both.
        """
        for key, ctrl in self._lut_controllers.items():
            name = self._fallback_channel_name(key)
            for view in ctrl.lut_views:
                view.set_fallback_name(name)

    def _update_lut_visibility(self, mode: ChannelMode) -> None:
        """Update LUT view visibility based on channel mode."""
        for lut_ctrl in self._lut_controllers.values():
            for view in lut_ctrl.lut_views:
                if lut_ctrl.key is None:
                    view.set_visible(mode == ChannelMode.GRAYSCALE)
                elif lut_ctrl.key == "RGB":
                    view.set_visible(mode == ChannelMode.RGBA)
                else:
                    view.set_visible(mode in {ChannelMode.COLOR, ChannelMode.COMPOSITE})

        # Mirror visibility on the shared histogram
        if hist := self._shared_histogram:
            for lut_ctrl in self._lut_controllers.values():
                key = lut_ctrl.key
                if key is None:
                    hist.set_channel_visible(key, mode == ChannelMode.GRAYSCALE)
                elif key == "RGB":
                    hist.set_channel_visible(key, mode == ChannelMode.RGBA)
                else:
                    visible = mode in {ChannelMode.COLOR, ChannelMode.COMPOSITE}
                    # Also respect the model's own visibility flag
                    hist.set_channel_visible(
                        key, visible and lut_ctrl.lut_model.visible
                    )

    def _is_rgba_compatible(self, resolved: ResolvedDisplayState) -> bool:
        # By design, RGBA channel display is only exposed for 2D image views.
        # 3D views use volume rendering and do not support this RGBA path.
        if len(resolved.visible_axes) != 2:
            return False
        return resolved.rgba_channel_count in {3, 4}

    @staticmethod
    def _rgba_fallback_mode(resolved: ResolvedDisplayState) -> ChannelMode:
        if resolved.channel_axis is not None:
            return ChannelMode.COMPOSITE
        return ChannelMode.GRAYSCALE

    def _prepare_channel_mode(self, resolved: ResolvedDisplayState) -> bool:
        """Update mode availability and coerce invalid mode selections.

        Returns True when this resolve pass can continue to `_apply_changes`.
        Returns False when mode coercion triggered a new model event.
        """
        rgba_compatible = self._is_rgba_compatible(resolved)
        self._view.set_channel_mode_enabled(ChannelMode.RGBA, rgba_compatible)
        if self._display_model.channel_mode != ChannelMode.RGBA or rgba_compatible:
            return True

        self._display_model.channel_mode = fb = self._rgba_fallback_mode(resolved)
        warnings.warn(
            "Cannot use RGBA mode for this data slice "
            f"(effective channel count is {resolved.rgba_channel_count}, "
            f"expected 3 or 4). Falling back to {fb.value}.",
            stacklevel=2,
        )
        return False

    # ------------------ Model callbacks ------------------

    def _fully_synchronize_view(self) -> None:
        """Reset and re-synchronize everything from scratch."""
        # Push channel mode even without data (e.g. display_model set before data)
        self._view.set_channel_mode(self._display_model.channel_mode)

        self._focused_slider_axis = None
        self._resolved = EMPTY_STATE
        if self._data_wrapper is not None:
            with self._view.currentIndexChanged.blocked():
                self._view.create_sliders(self._data_wrapper.coords)
        self._re_resolve()

        for lut_ctr in self._lut_controllers.values():
            lut_ctr.synchronize()
        self._push_fallback_channel_names()
        self._synchronize_roi()

    def _on_dims_changed(self) -> None:
        """Update sliders and info when dimension sizes change."""
        if self._data_wrapper is None:
            return
        with self._view.currentIndexChanged.blocked():
            self._view.create_sliders(self._data_wrapper.coords)
        self._re_resolve()

    def _synchronize_roi(self) -> None:
        """Fully re-synchronize the ROI view with the model."""
        if self.roi is not None:
            self._on_roi_model_bounding_box_changed(self.roi.bounding_box)
            self._on_roi_model_visible_changed(self.roi.visible)

    def _on_roi_model_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        if self._roi_view is not None:
            world_min = self._data_point_to_world(*bb[0])
            world_max = self._data_point_to_world(*bb[1])
            self._roi_view.set_bounding_box(world_min, world_max)

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
            while lut_ctrl.handles:
                handle = lut_ctrl.handles.pop()
                # disconnect model signals
                handle.model = None
                handle.remove()
                # handles are also added as lut_views via add_handle();
                # remove them so old GPU textures can be garbage-collected
                with suppress(ValueError):
                    lut_ctrl.lut_views.remove(handle)

    # ------------------ View callbacks ------------------

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self._display_model.current_index.update(self._view.current_index())

    def _on_view_ndim_toggle_requested(self, is_3d: bool) -> None:
        """Handle ndim toggle from view."""
        if self._data_wrapper is None:
            return
        current = self._resolved.visible_axes
        if not current:
            return
        if is_3d and len(current) == 2:
            z_ax = self._data_wrapper.guess_z_axis()
            if z_ax is None or z_ax in current:
                z_ax = next(
                    (
                        ax
                        for ax in reversed(self._resolved.data_coords)
                        if ax not in current
                    ),
                    None,
                )
            if z_ax is None:
                return
            self._display_model.visible_axes = (z_ax, *current)
        elif not is_3d and len(current) > 2:
            self._display_model.visible_axes = current[-2:]  # type: ignore[assignment]

    def _on_view_reset_zoom_clicked(self) -> None:
        """Reset the zoom level of the canvas."""
        self._canvas.set_range()

    def _on_roi_view_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        if self._roi_model:
            data_min = self._world_point_to_data(*bb[0])
            data_max = self._world_point_to_data(*bb[1])
            self._roi_model.bounding_box = (data_min, data_max)

    def _on_canvas_mouse_moved(self, event: MouseMoveEvent) -> None:
        """Respond to a mouse move event in the view."""
        x, y, _z = self._canvas.canvas_to_world((event.x, event.y))
        self._highlight_pos = (x, y)

        # update highlight display
        data_pos, channel_values = self._get_values_at_world_point(*self._highlight_pos)
        self._highlight_values(channel_values, data_pos)

    def _on_canvas_mouse_left(self) -> None:
        """Respond to a mouse leaving the canvas in the view."""
        self._highlight_pos = None
        self._highlight_values({}, self._highlight_pos)

    def _on_key_pressed(self, event: KeyPressEvent) -> None:
        handle_key_press(event, self)

    def _on_view_channel_mode_changed(self, mode: ChannelMode) -> None:
        self._display_model.channel_mode = mode

    # ------------------ Helper methods ------------------

    def _highlight_values(
        self,
        channel_values: dict[ChannelKey, float],
        data_pos: tuple[int, int] | None = None,
    ) -> None:
        """Highlights the given values for each channel."""
        # Update highlight each histogram. If the histogram channel is not present
        # in channel_values, the highlight will be set to None (i.e. hidden)
        for ch, hist in self._histograms.items():
            hist.highlight(channel_values.get(ch, None))

        # Also forward to shared histogram
        if self._shared_histogram is not None:
            self._shared_histogram.highlight(channel_values)

        if not channel_values:
            # clear hover info if no values found
            self._view.set_hover_info("")
        else:
            if data_pos is not None:
                pos = f"[{data_pos[0]}, {data_pos[1]}] "
            else:
                pos = " "  # pragma: no cover

            vals = []
            for ch, value in channel_values.items():
                fval = f"{value:.2f}".rstrip("0").rstrip(".")
                fch = f"{ch}: " if ch is not None else ""
                vals.append(f"{fch}{fval}")

            self._view.set_hover_info(pos + ",".join(vals))

    # The request cycle looks like this:
    # 1. something changes on the model requiring new data
    # 2. _request_data is called
    # 3. build_slice_requests returns request objects
    # 4. each request is submitted as a future
    # 5. when the future resolves, `_on_data_response_ready` draws the response.

    def _request_data(self) -> None:
        """Fetch and update the displayed data.

        This is called (frequently) when anything changes that requires a redraw.
        It fetches the current data slice from the model and updates the image handle.
        """
        if not self._data_wrapper:
            return  # pragma: no cover

        self._cancel_futures()
        self._current_gen = gen = next(self._gen_counter)

        for req in build_slice_requests(self._resolved, self._data_wrapper):
            future: Future[DataResponse]
            if self._async:
                future = _app.submit_task(process_request, req)
            else:
                future = Future()
                future.set_result(process_request(req))
            self._futures[future] = gen
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
            f, _ = self._futures.popitem()
            f.cancel()
        self._viewer_model.show_progress_spinner = False

    @_app.ensure_main_thread
    def _on_data_response_ready(self, future: Future[DataResponse]) -> None:
        # NOTE: popping the future is important because it holds a reference to
        # this controller in _done_callbacks, which would prevent GC otherwise.
        gen = self._futures.pop(future, -1)
        if not self._futures:
            self._viewer_model.show_progress_spinner = False

        if future.cancelled() or gen != self._current_gen:
            return

        try:
            response = future.result()
        except Exception as e:
            warnings.warn(f"Error fetching data: {e}", stacklevel=1)
            return

        for key, data in response.data.items():
            if data.size == 0:
                continue
            if (lut_ctrl := self._lut_controllers.get(key)) is None:
                if key is None:
                    model = self._display_model.default_lut
                elif key in self._display_model.luts:
                    model = self._display_model.luts[key]
                else:
                    # we received a new channel key that has not been set in the model
                    # so we create a new LUT model for it
                    model = self._display_model.luts[key] = LUTModel()

                lut_views = [self._view.add_lut_view(key)]
                if hist := self._histograms.get(key, None):
                    lut_views.append(hist)
                self._lut_controllers[key] = lut_ctrl = ChannelController(
                    key=key,
                    lut_model=model,
                    views=lut_views,
                )
                self._update_channel_dtype(key)
                fallback = self._fallback_channel_name(key)
                for v in lut_ctrl.lut_views:
                    v.set_fallback_name(fallback)
                # Connect new channel to shared histogram if it exists
                if self._shared_histogram is not None:
                    self._connect_shared_histogram_channel(key, lut_ctrl)

            if not lut_ctrl.handles:
                # we don't yet have any handles for this channel
                if response.n_visible_axes == 2:
                    handle = self._canvas.add_image(data)
                    lut_ctrl.add_handle(handle)
                elif response.n_visible_axes == 3:
                    handle = self._canvas.add_volume(data)
                    lut_ctrl.add_handle(handle)
                self._canvas.set_scales(self._resolved.visible_scales)

            sig_bits = wrp.significant_bits if (wrp := self._data_wrapper) else None
            has_broadcast = len(self.stats_updated) > 0
            stats = lut_ctrl.update_texture_data(
                data,
                need_histogram=has_broadcast,
                significant_bits=sig_bits,
            )
            if has_broadcast and stats is not None:
                self.stats_updated.emit(key, stats)

        self._canvas.refresh()
        # update highlight display
        if self._highlight_pos is not None:
            data_pos, channel_values = self._get_values_at_world_point(
                *self._highlight_pos
            )
            self._highlight_values(channel_values, data_pos)

    def _world_to_data(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x, y) to data (row, col) indices using visible scales."""
        scales = self._resolved.visible_scales
        if len(scales) >= 2:
            sx, sy = scales[-1], scales[-2]
            data_x = int(x / sx) if sx != 0 else int(x)
            data_y = int(y / sy) if sy != 0 else int(y)
        else:
            data_x, data_y = int(x), int(y)
        return data_y, data_x

    def _world_point_to_data(self, x: float, y: float) -> tuple[float, float]:
        """Convert world (x, y) to data (x, y) as floats using visible scales."""
        scales = self._resolved.visible_scales
        if len(scales) >= 2:
            sx, sy = scales[-1], scales[-2]
            data_x = x / sx if sx != 0 else x
            data_y = y / sy if sy != 0 else y
        else:
            data_x, data_y = x, y
        return data_x, data_y

    def _data_point_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Convert data (x, y) to world (x, y) using visible scales."""
        scales = self._resolved.visible_scales
        if len(scales) >= 2:
            sx, sy = scales[-1], scales[-2]
            return x * sx, y * sy
        return x, y

    def _get_values_at_world_point(
        self, x: float, y: float
    ) -> tuple[tuple[int, int], dict[ChannelKey, float]]:
        """Return (data_pos, channel_values) for world point (x, y)."""
        # TODO: handle 3D data
        n_vis = len(self._resolved.visible_axes)
        if n_vis != 2:  # pragma: no cover
            return (0, 0), {}

        data_y, data_x = self._world_to_data(x, y)

        if data_x < 0 or data_y < 0:
            return (data_y, data_x), {}

        values: dict[ChannelKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((data_y, data_x))) is not None:
                # Handle RGB
                if key == "RGB" and isinstance(value, np.ndarray):
                    values["R"] = value[0]
                    values["G"] = value[1]
                    values["B"] = value[2]
                    if value.shape[0] > 3:
                        values["A"] = value[3]
                else:
                    values[key] = cast("float", value)

        return (data_y, data_x), values


class _SharedHistogramLink:
    """Binds one ChannelController to a SharedHistogramCanvas.

    All signal connections use bound methods, so psygnal can weak-ref them
    and they can be cleanly disconnected via `disconnect()`.
    """

    def __init__(
        self,
        key: ChannelKey,
        ctrl: ChannelController,
        hist: SharedHistogramCanvas,
        fallback_name: str = "",
        significant_bits: int | None = None,
    ) -> None:
        self._key = key
        self._ctrl = ctrl
        self._hist = hist
        model = ctrl.lut_model

        ctrl.stats_updated.connect(self._on_stats)
        ctrl.clims_resolved.connect(self._on_clims_resolved)
        model.events.cmap.connect(self._on_cmap)
        model.events.visible.connect(self._on_visible)
        model.events.gamma.connect(self._on_gamma)
        model.events.name.connect(self._on_name)
        model.events.clim_bounds.connect(self._on_clim_bounds)

        # Set initial state
        hist.set_channel_color(key, model.cmap.color_stops[-1].color.rgba)
        hist.set_channel_visible(key, model.visible)
        hist.set_channel_gamma(key, model.gamma)
        hist.set_channel_name(key, model.name or fallback_name)
        if model.clim_bounds != (None, None):
            hist.set_clim_bounds(model.clim_bounds)
        if ctrl._last_clims is not None:
            hist.set_channel_clims(key, ctrl._last_clims)

        if handles := self._ctrl.handles:
            self._ctrl.update_texture_data(
                handles[0].data(), significant_bits=significant_bits
            )

    def _on_stats(self, stats: ImageStats) -> None:
        if stats.counts is not None and stats.bin_edges is not None:
            self._hist.set_channel_data(self._key, stats.counts, stats.bin_edges)

    def _on_clims_resolved(self, clims: tuple[float, float]) -> None:
        self._hist.set_channel_clims(self._key, clims)

    def _on_cmap(self, cmap: cmap_mod.Colormap) -> None:
        self._hist.set_channel_color(self._key, cmap.color_stops[-1].color.rgba)

    def _on_visible(self, visible: bool) -> None:
        self._hist.set_channel_visible(self._key, visible)

    def _on_gamma(self, gamma: float) -> None:
        self._hist.set_channel_gamma(self._key, gamma)

    def _on_name(self, name: str) -> None:
        self._hist.set_channel_name(self._key, name)

    def _on_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        self._hist.set_clim_bounds(bounds)

    def disconnect(self) -> None:
        model = self._ctrl.lut_model
        self._ctrl.stats_updated.disconnect(self._on_stats)
        self._ctrl.clims_resolved.disconnect(self._on_clims_resolved)
        model.events.cmap.disconnect(self._on_cmap)
        model.events.visible.disconnect(self._on_visible)
        model.events.gamma.disconnect(self._on_gamma)
        model.events.name.disconnect(self._on_name)
        model.events.clim_bounds.disconnect(self._on_clim_bounds)
