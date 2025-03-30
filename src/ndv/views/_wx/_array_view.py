from __future__ import annotations

import warnings
from pathlib import Path
from sys import version_info
from typing import TYPE_CHECKING, Any, cast

import psygnal
import wx
import wx.adv
import wx.lib.newevent
import wx.svg
from psygnal import EmissionInfo, Signal
from pyconify import svg_path

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsMinMax
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views._wx._labeled_slider import WxLabeledSlider
from ndv.views.bases import ArrayView, LutView

from .range_slider import RangeSlider

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap

    from ndv._types import AxisKey, ChannelKey
    from ndv.models._data_display_model import _ArrayDataDisplayModel


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
    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self.visible = wx.CheckBox(self, label="Visible")
        self.visible.SetValue(True)

        # Placeholder for the custom colormap combo box
        self.cmap = wx.ComboBox(
            self, choices=["gray", "green", "magenta"], style=wx.CB_DROPDOWN
        )

        # Placeholder for the QLabeledRangeSlider equivalent
        self.clims = RangeSlider(self, style=wx.SL_HORIZONTAL)
        self.clims.SetMax(65000)
        self.clims.SetValue(0, 65000)

        self.auto_clim = wx.ToggleButton(self, label="Auto", size=(50, -1))

        self.histogram = wx.ToggleButton(self, label="Hist", size=(40, -1))
        _add_icon(self.histogram, "foundation:graph-bar")

        # Layout
        self._widget_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._widget_sizer.Add(self.visible, 0, wx.ALL, 2)
        self._widget_sizer.Add(self.cmap, 0, wx.ALL, 2)
        self._widget_sizer.Add(self.clims, 1, wx.ALL, 2)
        self._widget_sizer.Add(self.auto_clim, 0, wx.ALL, 2)
        self._widget_sizer.Add(self.histogram, 0, wx.ALL, 2)
        self._widget_sizer.SetSizeHints(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self._widget_sizer, 0, wx.EXPAND, 5)

        self.SetSizer(self.sizer)
        self.Layout()


class WxLutView(LutView):
    # NB: In practice this will be a ChannelKey but Unions not allowed here.
    histogramRequested = psygnal.Signal(object)

    def __init__(self, parent: wx.Window, channel: ChannelKey = None) -> None:
        super().__init__()
        self._wxwidget = wdg = _WxLUTWidget(parent)
        self._channel = channel
        # TODO: Fix type
        self._histogram: Any | None = None
        # TODO: use emit_fast
        wdg.visible.Bind(wx.EVT_CHECKBOX, self._on_visible_changed)
        wdg.cmap.Bind(wx.EVT_COMBOBOX, self._on_cmap_changed)
        wdg.clims.Bind(wx.EVT_SLIDER, self._on_clims_changed)
        wdg.auto_clim.Bind(wx.EVT_TOGGLEBUTTON, self._on_autoscale_changed)
        wdg.histogram.Bind(wx.EVT_TOGGLEBUTTON, self._on_histogram_requested)

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

    def _on_autoscale_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            if self._wxwidget.auto_clim.GetValue():
                self._model.clims = ClimsMinMax()
            else:  # Manually scale
                clims = self._wxwidget.clims.GetValues()
                self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_histogram_requested(self, event: wx.CommandEvent) -> None:
        toggled = self._wxwidget.histogram.GetValue()
        if self._histogram:
            self._show_histogram(toggled)
        elif toggled:
            self.histogramRequested.emit(self._channel)

    def _add_histogram(self, histogram: wx.Window) -> None:
        # Setup references to the histogram
        self._histogram = histogram
        self._wxwidget.sizer.Add(histogram, 1, wx.ALIGN_CENTER, 5)

        # Assign a fixed size
        hist_size = wx.Size(self._wxwidget.Size.width, 100)
        histogram.SetSize(hist_size)
        histogram.SetMinSize(hist_size)

        # Show the histogram
        self._show_histogram(True)

    def _show_histogram(self, show: bool = True) -> None:
        if hist := cast("wx.Window", self._histogram):
            # Display the histogram
            hist.Show(show)

            # Resize the widget around the histogram
            # FIXME: Is all of this really necessary?
            size = wx.Size(self._wxwidget.Size)
            size.height += 100 if show else -100
            self._wxwidget.SetSize(size)
            self._wxwidget.SetMinSize(size)
            self._wxwidget.GetParent().Layout()

    # Public Methods
    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def set_channel_name(self, name: str) -> None:
        self._wxwidget.visible.SetLabel(name)

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        self._wxwidget.auto_clim.SetValue(not policy.is_manual)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        name = cmap.name.split(":")[-1]  # FIXME: this is a hack
        self._wxwidget.cmap.SetValue(name)

    def set_clims(self, clims: tuple[float, float]) -> None:
        # Block signals from changing clims
        with wx.EventBlocker(self._wxwidget.clims, wx.EVT_SLIDER.typeId):
            self._wxwidget.clims.SetValue(*clims)
            wx.SafeYield()

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
        self._wxwidget._widget_sizer.Insert(1, lbl, 0, wx.ALIGN_CENTER_VERTICAL, 5)
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
        data_model: _ArrayDataDisplayModel,
        viewer_model: ArrayViewerModel,
        parent: wx.Window = None,
    ) -> None:
        self._data_model = data_model
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
        if len(self._visible_axes) > 2:
            if not is_3d:  # is now 2D
                self._visible_axes = self._visible_axes[-2:]
        else:
            z_ax = None
            if wrapper := self._data_model.data_wrapper:
                z_ax = wrapper.guess_z_axis()
            if z_ax is None:
                # get the last slider that is not in visible axes
                sld = reversed(self._wxwidget.dims_sliders._sliders)
                z_ax = next(ax for ax in sld if ax not in self._visible_axes)
            self._visible_axes = (z_ax, *self._visible_axes)
        # TODO: a future PR may decide to set this on the model directly...
        # since we now have access to it.
        self.visibleAxesChanged.emit()

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
        view = WxRGBView(wdg, channel) if channel == "RGB" else WxLutView(wdg, channel)
        self._wxwidget.luts.Add(view._wxwidget, 0, wx.EXPAND | wx.BOTTOM, 5)
        self._luts[channel] = view
        # TODO: Reusable synchronization with ViewerModel
        view._wxwidget.histogram.Show(self._viewer_model.show_histogram_button)
        view.histogramRequested.connect(self.histogramRequested)

        # Update the layout to reflect above changes
        self._wxwidget.Layout()
        return view

    # TODO: Fix type
    def add_histogram(self, channel: ChannelKey, widget: Any) -> None:
        if lut := self._luts.get(channel, None):
            # FIXME: pygfx backend needs this to be widget._subwidget
            if hasattr(widget, "_subwidget"):
                widget = widget._subwidget

            # FIXME: Rendercanvas may make this unnecessary
            if (parent := widget.GetParent()) and parent is not self:
                widget.Reparent(lut._wxwidget)  # Reparent widget to this frame
                if parent:
                    parent.Destroy()
                widget.Show()

            # Add the histogram widget on the LUT
            lut._add_histogram(widget)

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
                lut._wxwidget.histogram.Show(value)
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
