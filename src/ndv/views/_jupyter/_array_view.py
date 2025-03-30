from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import psygnal

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsMinMax
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views.bases import ArrayView, LutView

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Iterator, Mapping, Sequence

    import cmap
    from psygnal import EmissionInfo
    from traitlets import HasTraits
    from vispy.app.backends import _jupyter_rfb

    from ndv._types import AxisKey, ChannelKey
    from ndv.models._data_display_model import _ArrayDataDisplayModel

# not entirely sure why it's necessary to specifically annotat signals as : PSignal
# i think it has to do with type variance?


@contextlib.contextmanager
def notifications_blocked(
    obj: HasTraits, name: str = "value", type: str = "change"
) -> Iterator[None]:
    # traitlets doesn't provide a public API for this
    notifiers: list | None = obj._trait_notifiers.get(name, {}).pop(type, None)
    try:
        yield
    finally:
        if notifiers is not None:
            obj._trait_notifiers[name][type] = notifiers


class JupyterLutView(LutView):
    # NB: In practice this will be a ChannelKey but Unions not allowed here.
    histogramRequested = psygnal.Signal(object)

    def __init__(self, channel: ChannelKey = None) -> None:
        self._channel = channel
        self._histogram_wdg: widgets.Widget | None = None
        # WIDGETS
        self._visible = widgets.Checkbox(value=True, indent=False)
        self._visible.layout.width = "60px"
        self._cmap = widgets.Dropdown(
            options=[
                "gray",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
                "viridis",
                "magma",
            ],
            value="gray",
        )
        self._cmap.layout.width = "200px"
        self._clims = widgets.FloatRangeSlider(
            value=[0, 2**16],
            min=0,
            max=2**16,
            step=1,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
        )
        self._clims.layout.width = "100%"
        self._auto_clim = widgets.ToggleButton(
            value=True,
            description="Auto",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Auto scale",
            layout=widgets.Layout(min_width="40px"),
        )
        self._histogram = widgets.ToggleButton(
            value=False,
            description="",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icon="bar-chart",
            tooltip="View Histogram",
            layout=widgets.Layout(width="40px"),
        )

        # LAYOUT

        lut_ctrls = widgets.HBox(
            [self._visible, self._cmap, self._clims, self._auto_clim, self._histogram]
        )
        self._histogram_container = widgets.HBox([])
        self.layout = widgets.VBox([lut_ctrls, self._histogram_container])

        # CONNECTIONS
        self._visible.observe(self._on_visible_changed, names="value")
        self._cmap.observe(self._on_cmap_changed, names="value")
        self._clims.observe(self._on_clims_changed, names="value")
        self._auto_clim.observe(self._on_autoscale_changed, names="value")
        self._histogram.observe(self._on_histogram_requested, names="value")

    # ------------------ emit changes to the controller ------------------

    def _on_clims_changed(self, change: dict[str, Any]) -> None:
        if self._model:
            clims = self._clims.value
            self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_visible_changed(self, change: dict[str, Any]) -> None:
        if self._model:
            self._model.visible = self._visible.value

    def _on_cmap_changed(self, change: dict[str, Any]) -> None:
        if self._model:
            self._model.cmap = self._cmap.value

    def _on_autoscale_changed(self, change: dict[str, Any]) -> None:
        if self._model:
            if change["new"]:  # Autoscale
                self._model.clims = ClimsMinMax()
            else:  # Manually scale
                clims = self._clims.value
                self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_histogram_requested(self, change: dict[str, Any]) -> None:
        if self._histogram_wdg:
            # show or hide the actual widget itself
            self._histogram_container.layout.display = (
                "flex" if change["new"] else "none"
            )
        else:
            self.histogramRequested.emit(self._channel)

    # ------------------ receive changes from the controller ---------------

    def set_channel_name(self, name: str) -> None:
        self._visible.description = name

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        self._auto_clim.value = not policy.is_manual

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        self._cmap.value = cmap.name.split(":")[-1]  # FIXME: this is a hack

    def set_clims(self, clims: tuple[float, float]) -> None:
        # block self._clims.observe, otherwise autoscale will be forced off
        with notifications_blocked(self._clims):
            self._clims.value = clims

    def set_channel_visible(self, visible: bool) -> None:
        self._visible.value = visible

    def set_gamma(self, gamma: float) -> None:
        pass

    def set_visible(self, visible: bool) -> None:
        # show or hide the actual widget itself
        self.layout.layout.display = "flex" if visible else "none"

    def close(self) -> None:
        self.layout.close()

    def frontend_widget(self) -> Any:
        return self.layout


class JupyterRGBView(JupyterLutView):
    def __init__(self, channel: ChannelKey = None) -> None:
        super().__init__(channel)
        self._cmap.layout.display = "none"


SPIN_GIF = str(Path(__file__).parent.parent / "_resources" / "spin.gif")


class JupyterArrayView(ArrayView):
    def __init__(
        self,
        canvas_widget: _jupyter_rfb.CanvasBackend,
        data_model: _ArrayDataDisplayModel,
        viewer_model: ArrayViewerModel,
    ) -> None:
        self._viewer_model = viewer_model
        self._viewer_model.events.connect(self._on_viewer_model_event)
        # WIDGETS
        self._data_model = data_model
        self._canvas_widget = canvas_widget
        self._visible_axes: Sequence[AxisKey] = []
        self._luts: dict[ChannelKey, JupyterLutView] = {}

        self._sliders: dict[Hashable, widgets.IntSlider] = {}
        self._slider_box = widgets.VBox([], layout=widgets.Layout(width="100%"))
        self._luts_box = widgets.VBox([], layout=widgets.Layout(width="100%"))

        # labels for data and hover info
        self._data_info_label = widgets.Label()
        self._hover_info_label = widgets.Label()

        # spinner to indicate progress
        self._progress_spinner = widgets.Image.from_file(
            SPIN_GIF, width=18, height=18, layout=widgets.Layout(display="none")
        )

        # the button that controls the display mode of the channels
        self._channel_mode_combo = widgets.Dropdown(
            options=[ChannelMode.GRAYSCALE, ChannelMode.COMPOSITE, ChannelMode.RGBA],
            value=str(ChannelMode.GRAYSCALE),
        )
        self._channel_mode_combo.layout.width = "120px"
        self._channel_mode_combo.layout.align_self = "flex-end"
        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

        # Reset zoom button
        self._reset_zoom_btn = widgets.Button(
            tooltip="Reset Zoom",
            icon="expand",
            layout=widgets.Layout(width="40px"),
        )
        self._reset_zoom_btn.on_click(self._on_reset_zoom_clicked)

        # 3d view button
        self._ndims_btn = widgets.ToggleButton(
            value=False,
            description="3D",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="View in 3D",
            layout=widgets.Layout(width="40px"),
        )
        self._ndims_btn.observe(self._on_ndims_toggled, names="value")

        # Add ROI button
        self._add_roi_btn = widgets.ToggleButton(
            value=False,
            description="New ROI",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Adds a new Rectangular ROI.",
            icon="square",
        )

        self._add_roi_btn.observe(self._on_add_roi_button_toggle, names="value")

        # LAYOUT

        top_row = widgets.HBox(
            [self._data_info_label, self._progress_spinner],
            layout=widgets.Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
            ),
        )

        try:
            width = getattr(canvas_widget, "css_width", "600px").replace("px", "")
            width = f"{int(width) + 4}px"
        except Exception:
            width = "604px"

        btns = widgets.HBox(
            [
                self._channel_mode_combo,
                self._ndims_btn,
                self._add_roi_btn,
                self._reset_zoom_btn,
            ],
            layout=widgets.Layout(justify_content="flex-end"),
        )
        self.layout = widgets.VBox(
            [
                top_row,
                self._canvas_widget,
                self._hover_info_label,
                self._slider_box,
                self._luts_box,
                btns,
            ],
            layout=widgets.Layout(width=width),
        )

        # CONNECTIONS

        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        sliders = []
        self._sliders.clear()
        for axis, _coords in coords.items():
            if not isinstance(_coords, range):
                raise NotImplementedError("Only range is supported for now")

            sld = widgets.IntSlider(
                value=_coords.start,
                min=_coords.start,
                max=_coords.stop - 1,
                step=_coords.step,
                description=str(axis),
                continuous_update=True,
                orientation="horizontal",
            )
            sld.layout.width = "99%"
            sld.observe(self._on_slider_change, "value")
            sliders.append(sld)
            self._sliders[axis] = sld
        self._slider_box.children = sliders

        self.currentIndexChanged.emit()

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        """Hide sliders based on visible axes."""
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                slider.layout.display = "none"
            elif show_remainder:
                slider.layout.display = "flex"

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        changed = False
        # this type ignore is only necessary because we had to override the signal
        # to be a PSignal in the class def above :(
        with self.currentIndexChanged.blocked():
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")

                if sld := self._sliders.get(axis):
                    if sld.value != val:
                        sld.value = val
                        changed = True
                else:  # pragma: no cover
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)
        if changed:
            self.currentIndexChanged.emit()

    def add_lut_view(self, channel: ChannelKey) -> JupyterLutView:
        """Add a LUT view to the viewer."""
        wdg = JupyterRGBView(channel) if channel == "RGB" else JupyterLutView(channel)
        layout = self._luts_box
        self._luts[channel] = wdg

        wdg.histogramRequested.connect(self.histogramRequested)
        layout.children = (*layout.children, wdg.layout)
        return wdg

    def remove_lut_view(self, view: LutView) -> None:
        """Remove a LUT view from the viewer."""
        view = cast("JupyterLutView", view)
        layout = self._luts_box
        layout.children = tuple(
            wdg for wdg in layout.children if wdg != view.frontend_widget()
        )

    def set_data_info(self, data_info: str) -> None:
        self._data_info_label.value = data_info

    def set_hover_info(self, hover_info: str) -> None:
        self._hover_info_label.value = hover_info

    def set_channel_mode(self, mode: ChannelMode) -> None:
        with self.channelModeChanged.blocked():
            self._channel_mode_combo.value = mode.value

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        """Emit signal when a slider value changes."""
        self.currentIndexChanged.emit()

    def _on_channel_mode_changed(self, change: dict[str, Any]) -> None:
        """Emit signal when the channel mode changes."""
        self.channelModeChanged.emit(ChannelMode(change["new"]))

    def add_histogram(self, channel: ChannelKey, widget: Any) -> None:
        if lut := self._luts.get(channel, None):
            # Resize widget to a respectable size
            widget.set_trait("css_height", "100px")
            # Add widget to view
            lut._histogram_container.children = (
                *lut._histogram_container.children,
                widget,
            )
            lut._histogram_wdg = widget

    def _on_add_roi_button_toggle(self, change: dict[str, Any]) -> None:
        """Emit signal when the channel mode changes."""
        self._viewer_model.interaction_mode = (
            InteractionMode.CREATE_ROI if change["new"] else InteractionMode.PAN_ZOOM
        )

    def remove_histogram(self, widget: Any) -> None:
        """Remove a histogram widget from the viewer."""

    def frontend_widget(self) -> Any:
        return self.layout

    def set_visible(self, visible: bool) -> None:
        # show or hide the actual widget itself
        from IPython import display

        if visible:
            display.display(self.layout)  # type: ignore [no-untyped-call]
        else:
            display.clear_output()  # type: ignore [no-untyped-call]

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._ndims_btn.value = len(axes) == 3

    def _on_ndims_toggled(self, change: dict[str, Any]) -> None:
        if len(self._visible_axes) > 2:
            if not change["new"]:  # is now 2D
                self._visible_axes = self._visible_axes[-2:]
        else:
            z_ax = None
            if wrapper := self._data_model.data_wrapper:
                z_ax = wrapper.guess_z_axis()
            if z_ax is None:
                # get the last slider that is not in visible axes
                z_ax = next(
                    ax for ax in reversed(self._sliders) if ax not in self._visible_axes
                )
            self._visible_axes = (z_ax, *self._visible_axes)
        # TODO: a future PR may decide to set this on the model directly...
        # since we now have access to it.
        self.visibleAxesChanged.emit()

    def _on_reset_zoom_clicked(self, change: dict[str, Any]) -> None:
        self.resetZoomClicked.emit()

    def close(self) -> None:
        self.layout.close()

    def _on_viewer_model_event(self, info: EmissionInfo) -> None:
        sig_name = info.signal.name
        value = info.args[0]
        if sig_name == "show_progress_spinner":
            self._progress_spinner.layout.display = "flex" if value else "none"
        elif sig_name == "interaction_mode":
            # If leaving CanvasMode.CREATE_ROI, uncheck the ROI button
            new, old = info.args
            if old == InteractionMode.CREATE_ROI:
                self._add_roi_btn.value = False
        elif sig_name == "show_histogram_button":
            # Note that "block" displays the icon better than "flex"
            for lut in self._luts.values():
                lut._histogram.layout.display = "block" if value else "none"
        elif sig_name == "show_roi_button":
            self._add_roi_btn.layout.display = "flex" if value else "none"
        elif sig_name == "show_channel_mode_selector":
            self._channel_mode_combo.layout.display = "flex" if value else "none"
        elif sig_name == "show_reset_zoom_button":
            self._reset_zoom_btn.layout.display = "flex" if value else "none"
        elif sig_name == "show_3d_button":
            self._ndims_btn.layout.display = "flex" if value else "none"
