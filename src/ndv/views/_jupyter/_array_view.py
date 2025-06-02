from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cmap
import ipywidgets as widgets
import psygnal
from IPython.display import Javascript, display

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsPercentile
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views.bases import ArrayView, LutView

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Iterator, Mapping, Sequence

    from psygnal import EmissionInfo
    from traitlets import HasTraits
    from vispy.app.backends import _jupyter_rfb

    from ndv._types import AxisKey, ChannelKey
    from ndv.views.bases._graphics._canvas import HistogramCanvas

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


class RightClickButton(widgets.ToggleButton):
    """Custom Button widget that shows a popup on right-click."""

    # TODO: These are likely unnecessary
    # _right_click_triggered = Bool(False).tag(sync=True)
    # popup_content = Unicode("Right-click menu").tag(sync=True)

    def __init__(self, channel: ChannelKey, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._channel = channel
        self.add_class(f"right-click-button{channel}")
        self.add_right_click_handler()

        self.lower_tail = widgets.BoundedFloatText(
            value=0.0,
            min=0.0,
            max=100.0,
            step=0.1,
            description="Ignore Lower Tail:",
            style={"description_width": "initial"},
        )
        self.upper_tail = widgets.BoundedFloatText(
            value=0.0,
            min=0.0,
            max=100.0,
            step=0.1,
            description="Ignore Upper Tail:",
            style={"description_width": "initial"},
        )
        self.popup_content = widgets.VBox(
            [self.lower_tail, self.upper_tail],
            layout=widgets.Layout(
                display="none",
            ),
        )
        self.popup_content.add_class(f"ipywidget-popup{channel}")
        display(self.popup_content)  # type: ignore [no-untyped-call]

    def add_right_click_handler(self) -> None:
        # fmt: off
        js_code = ( """
        (function() {
            function setup_rightclick() {
                // Get all buttons with the right-click-button class
                var button = document.getElementsByClassName(
                    'right-click-button""" + f"{self._channel}" + """'
                )[0];
                if (!button) {
                    return;
                }

                // For each button, add a contextmenu listener
                button.addEventListener('contextmenu', function(e) {
                    // Prevent default context menu
                    e.preventDefault();
                    e.stopPropagation();

                    // Get button position
                    var rect = this.getBoundingClientRect();
                    var scrollLeft = window.pageXOffset ||
                        document.documentElement.scrollLeft;
                    var scrollTop = window.pageYOffset ||
                        document.documentElement.scrollTop;

                    // Position the popup above the button
                    var popup = document.getElementsByClassName(
                        'ipywidget-popup""" + f"{self._channel}" + """'
                    )[0];
                    popup.style.display = '';
                    popup.style.position = 'absolute';
                    popup.style.top = (rect.bottom + scrollTop) + 'px';
                    popup.style.left = (rect.left + scrollLeft) + 'px';

                    // Style the popup
                    popup.style.background = 'white';
                    popup.style.border = '1px solid #ccc';
                    popup.style.borderRadius = '3px';
                    popup.style.padding = '8px';
                    popup.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                    popup.style.zIndex = '1000';

                    // Add to body
                    document.body.appendChild(popup);

                    // Close popup when clicking elsewhere
                    document.addEventListener('click', function closePopup(event) {
                        var popup = document.getElementsByClassName(
                            'ipywidget-popup""" + f"{self._channel}" + """'
                        )[0];
                        if (popup && !popup.contains(event.target)) {
                            popup.style.display = 'none';
                            document.removeEventListener('click', closePopup);
                        }
                    });

                    return false;
                });
            }

            // Make sure it works even after widget is redrawn/updated
            setTimeout(setup_rightclick, 1000);
        })();
        """)
        # fmt: on
        display(Javascript(js_code))  # type: ignore [no-untyped-call]


class JupyterLutView(LutView):
    # NB: In practice this will be a ChannelKey but Unions not allowed here.
    histogramRequested = psygnal.Signal(object)

    def __init__(
        self,
        channel: ChannelKey = None,
        default_luts: Sequence[Any] = ("gray", "green", "magenta", "red", "blue"),
    ) -> None:
        self._channel = channel
        self._histogram: HistogramCanvas | None = None
        # WIDGETS
        self._visible = widgets.Checkbox(value=True, indent=False)
        self._visible.layout.width = "60px"
        _luts = [cmap.Colormap(x).name.split(":")[-1] for x in default_luts]
        self._cmap = widgets.Dropdown(options=_luts, value=_luts[0])
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
        self._auto_clim = RightClickButton(
            channel=channel,
            value=True,
            description="Auto",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Auto scale",
            layout=widgets.Layout(min_width="40px"),
        )
        self._histogram_btn = widgets.ToggleButton(
            value=False,
            description="",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icon="bar-chart",
            tooltip="View Histogram",
            layout=widgets.Layout(width="40px"),
        )

        histogram_ctrl_width = 40
        self._log = widgets.ToggleButton(
            value=False,
            description="log",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Apply logarithm (base 10, count+1) to bins",
            layout=widgets.Layout(width=f"{histogram_ctrl_width}px"),
        )
        self._reset_histogram = widgets.Button(
            description="",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icon="expand",
            tooltip="Reset histogram view",
            layout=widgets.Layout(width=f"{histogram_ctrl_width}px"),
        )

        # LAYOUT

        lut_ctrls = widgets.HBox(
            [
                self._visible,
                self._cmap,
                self._clims,
                self._auto_clim,
                self._histogram_btn,
            ]
        )

        histogram_ctrls = widgets.VBox(
            [self._log, self._reset_histogram],
            layout=widgets.Layout(
                # Avoids scrollbar on buttons
                min_width=f"{histogram_ctrl_width + 10}px",
                # Floats buttons to the bottom
                justify_content="flex-end",
            ),
        )

        self._histogram_container = widgets.HBox(
            # Note that we'll add a histogram here later
            [histogram_ctrls],
            layout=widgets.Layout(
                # Constrains histogram to 100px tall
                max_height="100px",
                # Avoids vertical scrollbar from
                # histogram being *just a bit* too tall
                overflow="hidden",
                # Hide histogram initially
                display="none",
            ),
        )
        self.layout = widgets.VBox([lut_ctrls, self._histogram_container])

        # CONNECTIONS
        self._visible.observe(self._on_visible_changed, names="value")
        self._cmap.observe(self._on_cmap_changed, names="value")
        self._clims.observe(self._on_clims_changed, names="value")
        self._auto_clim.observe(self._on_autoscale_changed, names="value")
        self._auto_clim.lower_tail.observe(self._on_auto_tails_changed, names="value")
        self._auto_clim.upper_tail.observe(self._on_auto_tails_changed, names="value")
        self._histogram_btn.observe(self._on_histogram_requested, names="value")
        self._log.observe(self._on_log_toggled, names="value")
        self._reset_histogram.on_click(self._on_reset_histogram_clicked)

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
                lower_tail = self._auto_clim.lower_tail.value
                upper_tail = self._auto_clim.upper_tail.value
                self._model.clims = ClimsPercentile(
                    min_percentile=lower_tail, max_percentile=100 - upper_tail
                )
            else:  # Manually scale
                clims = self._clims.value
                self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_auto_tails_changed(self, change: dict[str, Any]) -> None:
        # Update clim policy if autoscaling is active
        if self._auto_clim.value:
            self._on_autoscale_changed({"new": True})

    def _on_histogram_requested(self, change: dict[str, Any]) -> None:
        # Generate the histogram if we haven't done so yet
        if not self._histogram:
            self.histogramRequested.emit(self._channel)
        # show or hide the histogram controls
        self._histogram_container.layout.display = "flex" if change["new"] else "none"

    def _on_log_toggled(self, change: dict[str, Any]) -> None:
        if hist := self._histogram:
            hist.set_log_base(10 if change["new"] else None)

    def _on_reset_histogram_clicked(self, change: dict[str, Any]) -> None:
        self._log.value = False
        if hist := self._histogram:
            hist.set_range()

    # ------------------ receive changes from the controller ---------------

    def set_channel_name(self, name: str) -> None:
        self._visible.description = name

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        with notifications_blocked(self._auto_clim):
            self._auto_clim.value = not policy.is_manual
            if isinstance(policy, ClimsPercentile):
                self._auto_clim.lower_tail.value = policy.min_percentile
                self._auto_clim.upper_tail.value = 100 - policy.max_percentile

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        self._cmap.value = cmap.name.split(":")[-1]  # FIXME: this is a hack

    def set_clims(self, clims: tuple[float, float]) -> None:
        # block self._clims.observe, otherwise autoscale will be forced off
        with notifications_blocked(self._clims):
            # FIXME: Internally the clims are being rounded to whole numbers.
            # The rounding is somehow avoiding notifications_blocked.
            self._clims.value = [int(c) for c in clims]

    def set_clim_bounds(
        self,
        bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        mi = 0 if bounds[0] is None else int(bounds[0])
        ma = 65535 if bounds[1] is None else int(bounds[1])
        # block self._clims.observe, otherwise autoscale will be forced off
        with notifications_blocked(self._clims):
            self._clims.min = mi
            self._clims.max = ma
            current_value = self._clims.value
            self._clims.value = (max(current_value[0], mi), min(current_value[1], ma))

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

    # ------------------ private methods ---------------

    def add_histogram(self, histogram: HistogramCanvas) -> None:
        widget = histogram.frontend_widget()
        # Resize widget to a respectable size
        widget.set_trait("css_height", "auto")
        # Add widget to view
        self._histogram_container.children = (
            *self._histogram_container.children,
            widget,
        )
        self._histogram = histogram


class JupyterRGBView(JupyterLutView):
    def __init__(self, channel: ChannelKey = None) -> None:
        super().__init__(channel)
        self._cmap.layout.display = "none"


SPIN_GIF = str(Path(__file__).parent.parent / "_resources" / "spin.gif")


class JupyterArrayView(ArrayView):
    def __init__(
        self,
        canvas_widget: _jupyter_rfb.CanvasBackend,
        viewer_model: ArrayViewerModel,
    ) -> None:
        self._viewer_model = viewer_model
        self._viewer_model.events.connect(self._on_viewer_model_event)
        # WIDGETS
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

        self._btns_box = widgets.HBox(
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
                self._btns_box,
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
        wdg = (
            JupyterRGBView(channel)
            if channel == "RGB"
            else JupyterLutView(channel, self._viewer_model.default_luts)
        )
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

    def add_histogram(self, channel: ChannelKey, histogram: HistogramCanvas) -> None:
        if lut := self._luts.get(channel, None):
            lut.add_histogram(histogram)

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
        self.nDimsRequested.emit(3 if change["new"] else 2)

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
                lut._histogram_btn.layout.display = "block" if value else "none"
        elif sig_name == "show_roi_button":
            self._add_roi_btn.layout.display = "flex" if value else "none"
        elif sig_name == "show_channel_mode_selector":
            self._channel_mode_combo.layout.display = "flex" if value else "none"
        elif sig_name == "show_reset_zoom_button":
            self._reset_zoom_btn.layout.display = "flex" if value else "none"
        elif sig_name == "show_3d_button":
            self._ndims_btn.layout.display = "flex" if value else "none"
        # elif sig_name == "show_play_button":
        #     ...
        elif sig_name == "show_data_info":
            self._data_info_label.display = "flex" if value else None
            self._hover_info_label.display = "flex" if value else None
        elif sig_name == "show_controls":
            # Show or hide the entire controls area (dims sliders + LUTs + btns)
            self._slider_box.display = "flex" if value else None
            self._btns_box.display = "flex" if value else None
            self._luts_box.display = "flex" if value else None
