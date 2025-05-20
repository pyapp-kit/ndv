from __future__ import annotations

import warnings
from pathlib import Path
from sys import version_info
from typing import TYPE_CHECKING, Any, cast

import cmap
import psygnal
import wx
import wx.adv
import wx.lib.newevent
import wx.svg
from psygnal import EmissionInfo, Signal
from pyconify import svg_path

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsPercentile
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views._wx._labeled_slider import WxLabeledSlider
from ndv.views.bases import ArrayView, LutView

from .range_slider import RangeSlider

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from ndv._types import AxisKey, ChannelKey
    from ndv.views.bases._graphics._canvas import HistogramCanvas


class _WxSpinner(wx.Panel):
    SPIN_GIF = str(Path(__file__).parent.parent / "_resources" / "spin.gif")

    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        # Load the animated GIF
        gif = wx.adv.Animation(self.SPIN_GIF)
        self.anim_ctrl = wx.adv.AnimationCtrl(self, -1, gif)

        # Set fixed size for the spinner
        self.SetSize((18, 18))
        self.anim_ctrl.SetSize((18, 18))

        # Start the animation
        self.anim_ctrl.Play()

        # Set semi-transparent effect (opacity not directly available in wx)
        # self.SetTransparent(153)  # 60% opacity (255 * 0.6)

        # Add the animation control to the sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.anim_ctrl, 0, wx.ALIGN_CENTER)
        self.SetSizer(sizer)


def _add_icon(btn: wx.AnyButton, icon: str) -> None:
    # Avoids https://github.com/urllib3/urllib3/issues/3020
    if version_info.minor < 10:
        return

    icon_path = svg_path(icon)
    bitmap = wx.BitmapBundle.FromSVGFile(str(icon_path), wx.Size(16, 16))
    # Note - label must be cleared first so bitmap is center-aligned
    btn.SetLabel("")
    btn.SetBitmapLabel(bitmap)


# mostly copied from _qt.qt_view._QLUTWidget
class _WxLUTWidget(wx.Panel):
    def __init__(self, parent: wx.Window, default_luts: Sequence[Any]) -> None:
        super().__init__(parent)

        # -- WDIDGETS -- #
        self.visible = wx.CheckBox(self, label="Visible")
        self.visible.SetValue(True)

        # Placeholder for the custom colormap combo box
        _luts = [cmap.Colormap(x).name.split(":")[-1] for x in default_luts]
        self.cmap = wx.ComboBox(self, choices=_luts, style=wx.CB_DROPDOWN)

        # Placeholder for the QLabeledRangeSlider equivalent
        self.clims = RangeSlider(self, style=wx.SL_HORIZONTAL)
        self.clims.SetMax(65000)
        self.clims.SetValue(0, 65000)

        self.auto_clim = wx.ToggleButton(self, label="Auto", size=(50, -1))

        self.auto_popup = wx.PopupTransientWindow(self, flags=wx.SIMPLE_BORDER)
        # FIXME: These TextCtrls do not seem to be editable
        # Seems related to its encapsulation in the popup window i.e. editable
        # when appended to e.g. self.lut_ctrls
        self.lower_tail = wx.SpinCtrlDouble(self.auto_popup)
        self.lower_tail.SetRange(0, 100)
        self.lower_tail.SetIncrement(0.1)
        self.upper_tail = wx.SpinCtrlDouble(self.auto_popup)
        self.upper_tail.SetRange(0, 100)
        self.upper_tail.SetIncrement(0.1)

        self._histogram_height = 100  # px
        self.histogram_btn = wx.ToggleButton(self, label="Hist", size=(40, -1))
        _add_icon(self.histogram_btn, "foundation:graph-bar")

        self.set_hist_range_btn = wx.Button(self, label="Reset", size=(40, -1))
        _add_icon(self.set_hist_range_btn, "fluent:full-screen-maximize-24-filled")
        self.set_hist_range_btn.Show(False)
        # Layout

        self.log_btn = wx.ToggleButton(self, label="Log", size=(40, -1))
        self.log_btn.SetToolTip("log (base 10, count+1)")
        _add_icon(self.log_btn, "mdi:math-log")
        self.log_btn.Show(False)

        # -- LAYOUT -- #

        # "main" lut controls (always visible)
        self.lut_ctrls = wx.BoxSizer(wx.HORIZONTAL)
        self.lut_ctrls.Add(self.visible, 0, wx.ALL, 2)
        self.lut_ctrls.Add(self.cmap, 0, wx.ALL, 2)
        self.lut_ctrls.Add(self.clims, 1, wx.ALL, 2)
        self.lut_ctrls.Add(self.auto_clim, 0, wx.ALL, 2)
        self.lut_ctrls.Add(self.histogram_btn, 0, wx.ALL, 2)

        # Autoscale popup
        self.autoscale_ctrls = wx.FlexGridSizer(rows=2, cols=2, hgap=0, vgap=0)
        lower_label = wx.StaticText(self.auto_popup, label="Exclude Darkest %")
        self.autoscale_ctrls.Add(lower_label, 0, wx.ALL, 2)
        self.autoscale_ctrls.Add(self.lower_tail, 0, wx.ALL, 2)
        upper_label = wx.StaticText(self.auto_popup, label="Exclude Brightest %")
        self.autoscale_ctrls.Add(upper_label, 0, wx.ALL, 2)
        self.autoscale_ctrls.Add(self.upper_tail, 0, wx.ALL, 2)

        self.auto_popup.SetSizer(self.autoscale_ctrls)
        self.auto_popup.Layout()
        self.autoscale_ctrls.Fit(self.auto_popup)

        # histogram controls go in their own sizer
        self.histogram_ctrls = wx.BoxSizer(wx.VERTICAL)
        self.histogram_ctrls.Add(self.log_btn, 0, wx.ALL, 2)
        self.histogram_ctrls.Add(self.set_hist_range_btn, 0, wx.ALL, 2)

        # histogram sizer contains controls + a histogram (which is added later)
        self._histogram_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._histogram_sizer.Add(self.histogram_ctrls, 0, wx.EXPAND, 5)

        # Overall layout
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.lut_ctrls, 0, wx.EXPAND, 5)
        self.sizer.Add(self._histogram_sizer, 0, wx.EXPAND, 5)
        self.sizer.SetSizeHints(self)

        self.SetSizer(self.sizer)
        self.Layout()


class WxLutView(LutView):
    # NB: In practice this will be a ChannelKey but Unions not allowed here.
    histogramRequested = psygnal.Signal(object)

    def __init__(
        self,
        parent: wx.Window,
        channel: ChannelKey = None,
        default_luts: Sequence[Any] = ("gray", "green", "magenta"),
    ) -> None:
        super().__init__()
        self._wxwidget = wdg = _WxLUTWidget(parent, default_luts)
        self._channel = channel
        self.histogram: HistogramCanvas | None = None
        # TODO: use emit_fast
        wdg.visible.Bind(wx.EVT_CHECKBOX, self._on_visible_changed)
        wdg.cmap.Bind(wx.EVT_COMBOBOX, self._on_cmap_changed)
        wdg.clims.Bind(wx.EVT_SLIDER, self._on_clims_changed)
        wdg.auto_clim.Bind(wx.EVT_TOGGLEBUTTON, self._on_autoscale_changed)
        wdg.auto_clim.Bind(wx.EVT_RIGHT_DOWN, self._on_autoscale_rclick)
        wdg.lower_tail.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_autoscale_tail_changed)
        wdg.upper_tail.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_autoscale_tail_changed)
        wdg.histogram_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_histogram_requested)
        wdg.log_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_log_btn_toggled)
        wdg.set_hist_range_btn.Bind(wx.EVT_BUTTON, self._on_set_histogram_range_clicked)

    # Event Handlers
    def _on_visible_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            self._model.visible = self._wxwidget.visible.GetValue()

    def _on_cmap_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            self._model.cmap = self._wxwidget.cmap.GetValue()

    def _on_clims_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            clims = self._wxwidget.clims.GetValues()
            self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_autoscale_rclick(self, event: wx.CommandEvent) -> None:
        btn = event.GetEventObject()
        pos = btn.ClientToScreen((0, 0))
        sz = btn.GetSize()
        self._wxwidget.auto_popup.Position(pos, (0, sz[1]))
        self._wxwidget.auto_popup.Popup()

    def _on_autoscale_tail_changed(self, event: wx.CommandEvent) -> None:
        self._on_autoscale_changed(event)

    def _on_autoscale_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            if self._wxwidget.auto_clim.GetValue():
                lower = self._wxwidget.lower_tail.GetValue()
                upper = self._wxwidget.upper_tail.GetValue()
                self._model.clims = ClimsPercentile(
                    min_percentile=lower, max_percentile=100 - upper
                )
            else:  # Manually scale
                clims = self._wxwidget.clims.GetValues()
                self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_histogram_requested(self, event: wx.CommandEvent) -> None:
        toggled = self._wxwidget.histogram_btn.GetValue()
        self._show_histogram(toggled)

        if self.histogram is None:
            self.histogramRequested.emit(self._channel)

    def _on_log_btn_toggled(self, event: wx.CommandEvent) -> None:
        toggled = self._wxwidget.log_btn.GetValue()
        if hist := self.histogram:
            hist.set_log_base(10 if toggled else None)

    def _on_set_histogram_range_clicked(self, event: wx.CommandEvent) -> None:
        # Reset log
        btn = self._wxwidget.log_btn
        if btn.GetValue():
            btn.SetValue(False)
            event = wx.PyCommandEvent(wx.EVT_TOGGLEBUTTON.typeId, btn.GetId())
            event.SetEventObject(btn)
            wx.PostEvent(btn.GetEventHandler(), event)
        if hist := self.histogram:
            hist.set_range()

    def _add_histogram(self, histogram: HistogramCanvas) -> None:
        widget = cast("wx.Window", histogram.frontend_widget())
        # FIXME: pygfx backend needs this to be widget._subwidget
        if hasattr(widget, "_subwidget"):
            widget = widget._subwidget

        # FIXME: Rendercanvas may make this unnecessary
        if (parent := widget.GetParent()) and parent is not self:
            widget.Reparent(self._wxwidget)  # Reparent widget to this frame
            wx.CallAfter(parent.Destroy)
            widget.Show()

        # Setup references to the histogram
        self.histogram = histogram

        # Assign a fixed size
        hist_size = wx.Size(self._wxwidget.Size.width, self._wxwidget._histogram_height)
        widget.SetMinSize(hist_size)
        self._wxwidget._histogram_sizer.Add(widget, 0, wx.ALIGN_CENTER, 5)

    def _show_histogram(self, show: bool = True) -> None:
        # Recursively show/hide _histograrm_sizer
        self._set_sizer_visibility(show, self._wxwidget._histogram_sizer)
        # Resize the widget
        size = wx.Size(self._wxwidget.lut_ctrls.MinSize)
        if show:
            size.height += self._wxwidget._histogram_height
        self._wxwidget.SetMinSize(size)
        self._wxwidget.GetParent().Layout()

    def _set_sizer_visibility(self, show: bool, sizer: wx.Sizer) -> None:
        for child in sizer.GetChildren():
            if child.IsSizer():
                self._set_sizer_visibility(show, child.GetSizer())
            elif child.IsWindow():
                child.GetWindow().Show(show)

    # Public Methods
    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def set_channel_name(self, name: str) -> None:
        self._wxwidget.visible.SetLabel(name)

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        self._wxwidget.auto_clim.SetValue(not policy.is_manual)
        if isinstance(policy, ClimsPercentile):
            self._wxwidget.lower_tail.SetValue(policy.min_percentile)
            self._wxwidget.upper_tail.SetValue(100 - policy.max_percentile)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        name = cmap.name.split(":")[-1]  # FIXME: this is a hack
        self._wxwidget.cmap.SetValue(name)

    def set_clims(self, clims: tuple[float, float]) -> None:
        # Block signals from changing clims
        with wx.EventBlocker(self._wxwidget.clims, wx.EVT_SLIDER.typeId):
            self._wxwidget.clims.SetValue(*clims)
            wx.SafeYield()

    def set_clim_bounds(
        self,
        bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        mi = 0 if bounds[0] is None else int(bounds[0])
        ma = 65535 if bounds[1] is None else int(bounds[1])
        self._wxwidget.clims.SetMin(mi)
        self._wxwidget.clims.SetMax(ma)

    def set_channel_visible(self, visible: bool) -> None:
        self._wxwidget.visible.SetValue(visible)

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._wxwidget.Show()
        else:
            self._wxwidget.Hide()

    def close(self) -> None:
        self._wxwidget.Close()


class WxRGBView(WxLutView):
    def __init__(self, parent: wx.Window, channel: ChannelKey = None) -> None:
        super().__init__(parent, channel)
        self._wxwidget.cmap.Hide()
        lbl = wx.StaticText(self._wxwidget, label="RGB")
        self._wxwidget.lut_ctrls.Insert(1, lbl, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        self._wxwidget.Layout()


# mostly copied from _qt.qt_view._QDimsSliders
class _WxDimsSliders(wx.Panel):
    currentIndexChanged = Signal()

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self._sliders: dict[AxisKey, WxLabeledSlider] = {}
        self.layout = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.layout)

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        for axis, _coords in coords.items():
            # Create a slider for axis if necessary
            if axis not in self._sliders:
                slider = WxLabeledSlider(self)
                slider.slider.Bind(wx.EVT_SLIDER, self._on_slider_changed)
                slider.label.SetLabel(str(axis))
                self.layout.Add(slider, 0, wx.EXPAND | wx.ALL, 5)
                self._sliders[axis] = slider

            # Update axis slider with coordinates
            slider = self._sliders[axis]
            if isinstance(_coords, range):
                slider.setRange(_coords.start, _coords.stop - 1)
                slider.setSingleStep(_coords.step)
            else:
                slider.setRange(0, len(_coords) - 1)

        self.currentIndexChanged.emit()

    def hide_dimensions(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                slider.Hide()
            elif show_remainder:
                slider.Show()

        self.Layout()

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        changed = False
        with self.currentIndexChanged.blocked():
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")
                if slider := self._sliders.get(axis):
                    if slider.value() != val:
                        changed = True
                        slider.setValue(val)
                else:
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)

        if changed:
            self.currentIndexChanged.emit()

    def _on_slider_changed(self, event: wx.CommandEvent) -> None:
        self.currentIndexChanged.emit()


class _WxArrayViewer(wx.Frame):
    def __init__(self, canvas_widget: wx.Window, parent: wx.Window = None):
        super().__init__(parent)

        # FIXME: pygfx backend needs this to be canvas_widget._subwidget
        if hasattr(canvas_widget, "_subwidget"):
            canvas_widget = canvas_widget._subwidget

        if (parent := canvas_widget.GetParent()) and parent is not self:
            canvas_widget.Reparent(self)  # Reparent canvas_widget to this frame
            if parent:
                parent.Destroy()
            canvas_widget.Show()

        self._canvas = canvas_widget

        # Dynamic sliders for dimensions
        self.dims_sliders = _WxDimsSliders(self)

        # Labels for data and hover information
        self._data_info_label = wx.StaticText(self, label="")
        self._hover_info_label = wx.StaticText(self, label="")

        # spinner to indicate progress
        self._progress_spinner = _WxSpinner(self)
        self._progress_spinner.Hide()

        # Channel mode combo box
        self.channel_mode_combo = wx.ComboBox(
            self,
            choices=[
                ChannelMode.GRAYSCALE.value,
                ChannelMode.COMPOSITE.value,
                ChannelMode.RGBA.value,
            ],
            style=wx.CB_DROPDOWN,
        )

        # Reset zoom button
        self.set_range_btn = wx.Button(self, label="Reset", size=(45, -1))
        self.set_range_btn.SetToolTip("Reset Zoom")
        _add_icon(self.set_range_btn, "fluent:full-screen-maximize-24-filled")

        # 3d view button
        self.ndims_btn = wx.ToggleButton(self, label="3D", size=(40, -1))

        # Add ROI button
        self.add_roi_btn = wx.ToggleButton(self, label="ROI", size=(40, -1))
        _add_icon(self.add_roi_btn, "mdi:vector-rectangle")

        # LUT layout (simple vertical grouping for LUT widgets)
        self.luts = wx.BoxSizer(wx.VERTICAL)

        self._btns = wx.BoxSizer(wx.HORIZONTAL)
        self._btns.AddStretchSpacer()
        self._btns.Add(self.channel_mode_combo, 0, wx.ALL, 4)
        self._btns.Add(self.set_range_btn, 0, wx.ALL, 4)
        self._btns.Add(self.ndims_btn, 0, wx.ALL, 4)
        self._btns.Add(self.add_roi_btn, 0, wx.ALL, 4)

        self._top_info = top_info = wx.BoxSizer(wx.HORIZONTAL)
        top_info.Add(self._data_info_label, 0, wx.EXPAND | wx.BOTTOM, 0)
        top_info.AddStretchSpacer()
        top_info.Add(self._progress_spinner, 0, wx.EXPAND | wx.BOTTOM, 0)

        inner = wx.BoxSizer(wx.VERTICAL)
        inner.Add(top_info, 0, wx.EXPAND | wx.BOTTOM, 5)
        inner.Add(self._canvas, 1, wx.EXPAND | wx.ALL)
        inner.Add(self._hover_info_label, 0, wx.EXPAND | wx.BOTTOM)
        inner.Add(self.dims_sliders, 0, wx.EXPAND | wx.BOTTOM)
        inner.Add(self.luts, 0, wx.EXPAND)
        inner.Add(self._btns, 0, wx.EXPAND)

        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(inner, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(outer)
        self.SetInitialSize(wx.Size(600, 800))
        self.Layout()


class WxArrayView(ArrayView):
    def __init__(
        self,
        canvas_widget: wx.Window,
        viewer_model: ArrayViewerModel,
        parent: wx.Window = None,
    ) -> None:
        self._viewer_model = viewer_model
        self._viewer_model.events.connect(self._on_viewer_model_event)
        self._wxwidget = wdg = _WxArrayViewer(canvas_widget, parent)
        # Mapping of channel key to LutViews
        self._luts: dict[ChannelKey, WxLutView] = {}
        self._visible_axes: Sequence[AxisKey] = []

        wdg.dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)
        wdg.channel_mode_combo.Bind(wx.EVT_COMBOBOX, self._on_channel_mode_changed)
        wdg.set_range_btn.Bind(wx.EVT_BUTTON, self._on_reset_zoom_clicked)
        wdg.ndims_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_ndims_toggled)
        wdg.add_roi_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_add_roi_toggled)

    def _on_channel_mode_changed(self, event: wx.CommandEvent) -> None:
        mode = self._wxwidget.channel_mode_combo.GetValue()
        self.channelModeChanged.emit(mode)

    def _on_reset_zoom_clicked(self, event: wx.CommandEvent) -> None:
        self.resetZoomClicked.emit()

    def _on_ndims_toggled(self, event: wx.CommandEvent) -> None:
        is_3d = self._wxwidget.ndims_btn.GetValue()
        self.nDimsRequested.emit(3 if is_3d else 2)

    def _on_add_roi_toggled(self, event: wx.CommandEvent) -> None:
        create_roi = self._wxwidget.add_roi_btn.GetValue()
        self._viewer_model.interaction_mode = (
            InteractionMode.CREATE_ROI if create_roi else InteractionMode.PAN_ZOOM
        )

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes  # no widget to control this yet

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._wxwidget.ndims_btn.SetValue(len(axes) == 3)

    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def add_lut_view(self, channel: ChannelKey) -> WxLutView:
        wdg = self.frontend_widget()
        view = (
            WxRGBView(wdg, channel)
            if channel == "RGB"
            else WxLutView(wdg, channel, self._viewer_model.default_luts)
        )
        self._wxwidget.luts.Add(view._wxwidget, 0, wx.EXPAND | wx.BOTTOM, 5)
        self._luts[channel] = view
        # TODO: Reusable synchronization with ViewerModel
        view._wxwidget.histogram_btn.Show(self._viewer_model.show_histogram_button)
        view.histogramRequested.connect(self.histogramRequested)

        # Update the layout to reflect above changes
        self._wxwidget.Layout()
        return view

    # TODO: Fix type
    def add_histogram(self, channel: ChannelKey, canvas: HistogramCanvas) -> None:
        if lut := self._luts.get(channel, None):
            # Add the histogram widget on the LUT
            lut._add_histogram(canvas)
        self._wxwidget.Layout()

    def remove_lut_view(self, lut: LutView) -> None:
        wxwdg = cast("_WxLUTWidget", lut.frontend_widget())
        self._wxwidget.luts.Detach(wxwdg)
        wxwdg.Destroy()
        self._wxwidget.Layout()

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        self._wxwidget.dims_sliders.create_sliders(coords)
        self._wxwidget.Layout()

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        self._wxwidget.dims_sliders.hide_dimensions(axes_to_hide, show_remainder)
        self._wxwidget.Layout()

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        return self._wxwidget.dims_sliders.current_index()

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        self._wxwidget.dims_sliders.set_current_index(value)

    def set_data_info(self, text: str) -> None:
        self._wxwidget._data_info_label.SetLabel(text)

    def set_hover_info(self, text: str) -> None:
        self._wxwidget._hover_info_label.SetLabel(text)

    def set_channel_mode(self, mode: ChannelMode) -> None:
        self._wxwidget.channel_mode_combo.SetValue(mode)

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._wxwidget.Show()
        else:
            self._wxwidget.Hide()

    def close(self) -> None:
        self._wxwidget.Close()

    def _on_viewer_model_event(self, info: EmissionInfo) -> None:
        sig_name = info.signal.name
        value = info.args[0]
        if sig_name == "show_progress_spinner":
            self._wxwidget._progress_spinner.Show(value)
            self._wxwidget._top_info.Layout()
        elif sig_name == "interaction_mode":
            # If leaving CanvasMode.CREATE_ROI, uncheck the ROI button
            new, old = info.args
            if old == InteractionMode.CREATE_ROI:
                self._wxwidget.add_roi_btn.SetValue(False)
        elif sig_name == "show_histogram_button":
            for lut in self._luts.values():
                lut._wxwidget.histogram_btn.Show(value)
                lut._wxwidget.Layout()
        elif sig_name == "show_roi_button":
            self._wxwidget.add_roi_btn.Show(value)
            self._wxwidget._btns.Layout()
        elif sig_name == "show_channel_mode_selector":
            self._wxwidget.channel_mode_combo.Show(value)
            self._wxwidget._btns.Layout()
        elif sig_name == "show_reset_zoom_button":
            self._wxwidget.set_range_btn.Show(value)
            self._wxwidget._btns.Layout()
        elif sig_name == "show_3d_button":
            self._wxwidget.ndims_btn.Show(value)
            self._wxwidget._btns.Layout()
