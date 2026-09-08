from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cmap
import psygnal
from anywidget.experimental import widget

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsPercentile
from ndv.models._viewer_model import InteractionMode
from ndv.views.bases import ArrayView, LUTView

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from psygnal import EmissionInfo

    from ndv._types import AxisKey, ChannelKey
    from ndv.models._viewer_model import ArrayViewerModel
    from ndv.views.bases._graphics._canvas import HistogramCanvas

_STATIC = Path(__file__).parent / "static"


def _cmap_css(cm: cmap.Colormap) -> str:
    """Extract the linear-gradient(...) value from cmap's CSS output."""
    line = cm.to_css(max_stops=16).split("\n")[1]
    return line.replace("background: ", "").rstrip(";")  # type: ignore[no-any-return]


_ESM_FILE = _STATIC / "ndv-jupyter.js"
_CSS_FILE = _STATIC / "style.css"


def _read_or_stub(path: Path) -> str:
    """Read file content, or return a stub if the JS hasn't been built yet."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "export function render({ el }) { el.textContent = 'JS not built'; }"


@widget(
    esm=_read_or_stub(_ESM_FILE),
    css=_read_or_stub(_CSS_FILE),
)
@dataclass
class NdvWidgetState:
    """Widget state synced between Python and JS via anywidget + psygnal."""

    events = psygnal.SignalGroupDescriptor()

    # Dimension sliders: [{axis, label, min, max, value, step, visible}]
    sliders: list[dict] = field(default_factory=list)

    # LUTs: [{key, name, visible, cmap_name, cmap_colors, cmap_options,
    #         clim_min, clim_max, clim_bound_min, clim_bound_max,
    #         auto_clim, auto_lower_tail, auto_upper_tail, gamma,
    #         show_histogram_btn, show_cmap, row_visible}]
    luts: list[dict] = field(default_factory=list)

    # Channel mode
    channel_mode: str = "grayscale"
    channel_mode_options: list[dict] = field(default_factory=list)

    # Info labels
    data_info: str = ""
    hover_info: str = ""

    # UI visibility flags (from ViewerModel)
    is_3d: bool = False
    show_3d_button: bool = True
    show_controls: bool = True
    show_channel_mode_selector: bool = True
    show_reset_zoom_button: bool = True
    show_roi_button: bool = False
    show_histogram_button: bool = True
    use_shared_histogram: bool = False
    show_data_info: bool = True
    progress_visible: bool = False

    # Shared histogram state (toggle buttons live in the controls JS)
    shared_histogram_visible: bool = False
    shared_histogram_log: bool = False

    # Theme info synced from JS (read-only on Python side)
    _theme_kind: str = "dark"
    _theme_background: str = ""

    # JS -> Python event channel. JS writes via model.set + save_changes.
    _js_event: dict = field(default_factory=dict)


class JupyterLUTView(LUTView):
    """Thin proxy -- translates LUTView ABC calls into NdvWidgetState.luts."""

    histogramRequested = psygnal.Signal(object)

    def __init__(
        self,
        parent: NdvWidgetState,
        channel: ChannelKey,
        default_luts: Sequence[Any] = ("gray", "green", "magenta", "red", "blue"),
    ) -> None:
        self._parent = parent
        self._channel = channel
        self._key_str = str(channel)
        self._is_rgb = channel == "RGB"
        self._default_luts = [str(x) for x in default_luts]

    def _update_lut_field(self, **kwargs: Any) -> None:
        luts = list(self._parent.luts)
        for i, lut in enumerate(luts):
            if lut["key"] == self._key_str:
                luts[i] = {**lut, **kwargs}
                break
        self._parent.luts = luts

    def set_channel_name(self, name: str) -> None:
        self._update_lut_field(name=name)

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        auto = not policy.is_manual
        fields: dict[str, Any] = {"auto_clim": auto}
        if isinstance(policy, ClimsPercentile):
            fields["auto_lower_tail"] = policy.min_percentile
            fields["auto_upper_tail"] = 100 - policy.max_percentile
        self._update_lut_field(**fields)

    def set_colormap(self, colormap: Any) -> None:
        if not isinstance(colormap, cmap.Colormap):
            colormap = cmap.Colormap(colormap)
        name = colormap.name.split(":")[-1]
        self._update_lut_field(cmap_name=name, cmap_css=_cmap_css(colormap))

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._update_lut_field(clim_min=clims[0], clim_max=clims[1])

    def set_clim_bounds(
        self, bounds: tuple[float | None, float | None] = (None, None)
    ) -> None:
        mi = 0.0 if bounds[0] is None else float(bounds[0])
        ma = 65535.0 if bounds[1] is None else float(bounds[1])
        self._update_lut_field(clim_bound_min=mi, clim_bound_max=ma)

    def set_channel_visible(self, visible: bool) -> None:
        self._update_lut_field(visible=visible)

    def set_gamma(self, gamma: float) -> None:
        self._update_lut_field(gamma=gamma)

    def set_visible(self, visible: bool) -> None:
        self._update_lut_field(row_visible=visible)

    def close(self) -> None:
        self._parent.luts = [
            lt for lt in self._parent.luts if lt["key"] != self._key_str
        ]

    def frontend_widget(self) -> Any:
        return self._parent


class JupyterArrayView(ArrayView):
    """ArrayView backed by anywidget, composing [canvas, controls, histogram]."""

    def __init__(
        self,
        canvas_widget: Any,
        viewer_model: ArrayViewerModel,
    ) -> None:
        self._viewer_model = viewer_model
        self._canvas_widget = canvas_widget
        self._visible_axes: Sequence[AxisKey] = []
        self._luts: dict[ChannelKey, JupyterLUTView] = {}
        self._current_index: dict[AxisKey, int] = {}
        self._shared_histogram: HistogramCanvas | None = None
        self._histogram_frontend: Any | None = None

        # Controls widget (sliders, LUTs, toolbar, info bar)
        self._widget = NdvWidgetState()

        # Sync initial viewer model flags
        self._sync_viewer_model_flags()
        self._viewer_model.events.connect(self._on_viewer_model_event)

        # Listen for JS events via _js_event field changes
        self._updating_channel_mode = False
        self._widget.events._js_event.connect(self._on_js_event)
        self._widget.events.channel_mode.connect(self._on_channel_mode_field_changed)
        self._widget.events.shared_histogram_log.connect(
            self._on_shared_histogram_log_changed
        )
        self._widget.events.shared_histogram_visible.connect(
            self._on_shared_histogram_visible_changed
        )

    # ---- JS event handler (JS -> Python via _js_event field) ----

    def _on_js_event(self, msg: dict) -> None:
        event_type = msg.get("type")
        if event_type == "slider_changed":
            axis_key = self._axis_from_str(msg["axis"])
            self._current_index[axis_key] = msg["value"]
            sliders = list(self._widget.sliders)
            for i, s in enumerate(sliders):
                if s["axis"] == msg["axis"]:
                    sliders[i] = {**s, "value": msg["value"]}
            self._widget.sliders = sliders
            self.currentIndexChanged.emit()

        elif event_type == "reset_zoom":
            self.resetZoomClicked.emit()

        elif event_type == "ndim_toggle":
            self.ndimToggleRequested.emit(msg["value"])

        elif event_type == "shared_histogram_request":
            self.sharedHistogramRequested.emit()

        elif event_type == "roi_toggle":
            self._viewer_model.interaction_mode = (
                InteractionMode.CREATE_ROI if msg["value"] else InteractionMode.PAN_ZOOM
            )

        elif event_type == "update_lut":
            key = self._key_from_str(msg["key"])
            lut_view = self._luts.get(key)
            if lut_view is None or lut_view._model is None:
                return
            model = lut_view._model

            if "visible" in msg:
                model.visible = msg["visible"]
            if "cmap_name" in msg:
                model.cmap = msg["cmap_name"]
            if "clim_min" in msg and "clim_max" in msg:
                model.clims = ClimsManual(min=msg["clim_min"], max=msg["clim_max"])
            if "auto_clim" in msg:
                if msg["auto_clim"]:
                    model.clims = ClimsPercentile(
                        min_percentile=msg.get("auto_lower_tail", 0),
                        max_percentile=100 - msg.get("auto_upper_tail", 0),
                    )
                else:
                    model.clims = ClimsManual(
                        min=msg.get("clim_min", 0),
                        max=msg.get("clim_max", 65535),
                    )

    def _on_shared_histogram_log_changed(self) -> None:
        if self._shared_histogram is not None:
            log_on = self._widget.shared_histogram_log
            self._shared_histogram.set_log_base(10 if log_on else None)

    def _on_channel_mode_field_changed(self) -> None:
        if not self._updating_channel_mode:
            self.channelModeChanged.emit(ChannelMode(self._widget.channel_mode))

    # ---- Axis key serialization ----

    def _axis_str_map(self) -> dict[str, AxisKey]:
        return {str(ax): ax for ax in self._current_index}

    def _axis_from_str(self, s: str) -> AxisKey:
        mapping = self._axis_str_map()
        if s in mapping:
            return mapping[s]
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    def _key_str_map(self) -> dict[str, ChannelKey]:
        return {str(k): k for k in self._luts}

    def _key_from_str(self, s: str) -> ChannelKey:
        mapping = self._key_str_map()
        if s in mapping:
            return mapping[s]
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    # ---- ArrayView ABC implementation ----

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        sliders = []
        self._current_index.clear()
        for axis, _coords in coords.items():
            if not isinstance(_coords, range):
                raise NotImplementedError("Only range is supported for now")
            self._current_index[axis] = _coords.start
            sliders.append(
                {
                    "axis": str(axis),
                    "label": str(axis),
                    "min": _coords.start,
                    "max": _coords.stop - 1,
                    "value": _coords.start,
                    "step": _coords.step,
                    "visible": True,
                }
            )
        self._widget.sliders = sliders
        self.currentIndexChanged.emit()

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        return dict(self._current_index)

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        changed = False
        with self.currentIndexChanged.blocked():
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")
                if self._current_index.get(axis) != val:
                    self._current_index[axis] = val
                    changed = True

            if changed:
                sliders = list(self._widget.sliders)
                for i, s in enumerate(sliders):
                    ax = self._axis_from_str(s["axis"])
                    if ax in value:
                        v = value[ax]
                        if not isinstance(v, slice):
                            sliders[i] = {**s, "value": v}
                self._widget.sliders = sliders

        if changed:
            self.currentIndexChanged.emit()

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._widget.is_3d = len(axes) > 2

    def set_channel_mode(self, mode: ChannelMode) -> None:
        self._updating_channel_mode = True
        self._widget.channel_mode = mode.value
        self._updating_channel_mode = False

    def set_channel_mode_enabled(self, mode: ChannelMode, enabled: bool) -> None:
        options = list(self._widget.channel_mode_options)
        for i, opt in enumerate(options):
            if opt["value"] == mode.value:
                options[i] = {**opt, "enabled": enabled}
                break
        self._widget.channel_mode_options = options

    def set_data_info(self, data_info: str) -> None:
        self._widget.data_info = data_info

    def set_hover_info(self, hover_info: str) -> None:
        self._widget.hover_info = hover_info

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = True
    ) -> None:
        sliders = list(self._widget.sliders)
        for i, s in enumerate(sliders):
            ax = self._axis_from_str(s["axis"])
            if ax in axes_to_hide:
                sliders[i] = {**s, "visible": False}
            elif show_remainder:
                sliders[i] = {**s, "visible": True}
        self._widget.sliders = sliders

    def add_lut_view(self, channel: ChannelKey) -> JupyterLUTView:
        lut_view = JupyterLUTView(
            self._widget, channel, self._viewer_model.default_luts
        )
        self._luts[channel] = lut_view

        key_str = str(channel)
        lut_options = []
        for x in self._viewer_model.default_luts:
            cm = cmap.Colormap(x)
            name = cm.name.split(":")[-1]
            lut_options.append({"name": name, "css": _cmap_css(cm)})
        new_lut: dict[str, Any] = {
            "key": key_str,
            "name": key_str,
            "visible": True,
            "cmap_name": lut_options[0]["name"] if lut_options else "gray",
            "cmap_css": "",
            "cmap_options": lut_options,
            "clim_min": 0,
            "clim_max": 65535,
            "clim_bound_min": 0,
            "clim_bound_max": 65535,
            "auto_clim": True,
            "auto_lower_tail": 0,
            "auto_upper_tail": 0,
            "gamma": 1.0,
            "show_histogram_btn": self._viewer_model.show_histogram_button,
            "show_cmap": channel != "RGB",
            "row_visible": True,
        }
        self._widget.luts = [*self._widget.luts, new_lut]
        return lut_view

    def remove_lut_view(self, view: LUTView) -> None:
        if not isinstance(view, JupyterLUTView):
            return
        self._luts.pop(view._channel, None)
        self._widget.luts = [
            lt for lt in self._widget.luts if lt["key"] != view._key_str
        ]

    def add_histogram(self, channel: ChannelKey, widget: Any) -> None:
        raise NotImplementedError("Per-channel histograms not implemented")

    def remove_histogram(self, widget: Any) -> None:
        raise NotImplementedError("Per-channel histograms not implemented")

    def set_histogram_widget(self, widget: Any) -> None:
        """Pre-set the histogram frontend widget (included in layout, hidden)."""
        self._histogram_frontend = widget
        if hasattr(widget, "css_height"):
            widget.css_height = "120px"
        if hasattr(widget, "css_width"):
            widget.css_width = "100%"
        # Also set ipywidgets layout so the widget container stretches
        if hasattr(widget, "layout"):
            widget.layout.width = "100%"
            widget.layout.height = "120px"

    def add_shared_histogram(self, widget: Any) -> None:
        self._shared_histogram = widget
        if self._histogram_frontend is None:
            self.set_histogram_widget(widget.frontend_widget())
        self._widget.shared_histogram_visible = True

    def remove_shared_histogram(self) -> None:
        self._shared_histogram = None
        self._widget.shared_histogram_visible = False

    def _on_shared_histogram_visible_changed(self) -> None:
        """Toggle histogram widget visibility when JS toggles the flag."""
        visible = self._widget.shared_histogram_visible
        self._set_histogram_visible(visible)

    def _set_histogram_visible(self, visible: bool) -> None:
        """Show or hide the histogram widget in the layout."""
        import ipywidgets

        box = getattr(self, "_histogram_box", None)
        if isinstance(box, ipywidgets.Box):
            box.layout.display = "" if visible else "none"

    def frontend_widget(self) -> Any:
        import ipywidgets
        from IPython import display as ipy_display

        # Wrap NdvWidgetState (descriptor-based, not an ipywidget) in Output
        # so it can be a child of VBox.
        controls_output = ipywidgets.Output()
        with controls_output:
            ipy_display.display(self._widget)  # type: ignore[no-untyped-call]

        # Histogram box: always in layout, hidden until toggled
        hist_children = []
        if self._histogram_frontend is not None:
            hist_children = [self._histogram_frontend]
        self._histogram_box = ipywidgets.Box(
            hist_children,
            layout=ipywidgets.Layout(display="none", width="100%", overflow="hidden"),
        )

        return ipywidgets.VBox(
            [self._canvas_widget, controls_output, self._histogram_box]
        )

    def set_visible(self, visible: bool) -> None:
        if visible:
            from IPython import display

            display.display(self.frontend_widget())  # type: ignore[no-untyped-call]

    def close(self) -> None:
        self._viewer_model.events.disconnect(self._on_viewer_model_event)
        self._widget.events._js_event.disconnect(self._on_js_event)
        self._widget.events.channel_mode.disconnect(self._on_channel_mode_field_changed)
        self._widget.events.shared_histogram_log.disconnect(
            self._on_shared_histogram_log_changed
        )
        self._widget.events.shared_histogram_visible.disconnect(
            self._on_shared_histogram_visible_changed
        )

    # ---- ViewerModel reactivity ----

    def _sync_viewer_model_flags(self) -> None:
        vm = self._viewer_model
        self._widget.show_3d_button = vm.show_3d_button
        self._widget.show_controls = vm.show_controls
        self._widget.show_channel_mode_selector = vm.show_channel_mode_selector
        self._widget.show_reset_zoom_button = vm.show_reset_zoom_button
        self._widget.show_roi_button = vm.show_roi_button
        self._widget.show_histogram_button = vm.show_histogram_button
        self._widget.use_shared_histogram = vm.use_shared_histogram
        self._widget.show_data_info = vm.show_data_info
        self._widget.progress_visible = vm.show_progress_spinner
        self._widget.channel_mode_options = [
            {"value": m.value, "label": m.value, "enabled": True}
            for m in [ChannelMode.GRAYSCALE, ChannelMode.COMPOSITE, ChannelMode.RGBA]
        ]

    def _on_viewer_model_event(self, info: EmissionInfo) -> None:
        sig_name = info.signal.name
        value = info.args[0]
        flag_map: dict[str, str] = {
            "show_progress_spinner": "progress_visible",
            "show_3d_button": "show_3d_button",
            "show_controls": "show_controls",
            "show_channel_mode_selector": "show_channel_mode_selector",
            "show_reset_zoom_button": "show_reset_zoom_button",
            "show_roi_button": "show_roi_button",
            "show_histogram_button": "show_histogram_button",
            "use_shared_histogram": "use_shared_histogram",
            "show_data_info": "show_data_info",
        }
        if sig_name in flag_map:
            setattr(self._widget, flag_map[sig_name], value)
        elif sig_name == "interaction_mode":
            pass  # handled by JS side
