from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import wx
import wx.adv
import wx.lib.newevent
import wx.svg
from psygnal import Signal
from pyconify import svg_path

from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsMinMax
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

        self.auto_clim = wx.ToggleButton(self, label="Auto")

        self.histogram = wx.ToggleButton(self)
        # FIXME: Polish this code, make reusable
        icon_path = svg_path("foundation:graph-bar")
        icon = wx.svg.SVGimage.CreateFromFile(str(icon_path))
        # NB 5 is a magic number for the margins
        bmp_side_len = self.auto_clim.Size.height - 5
        bmp_size = wx.Size(bmp_side_len, bmp_side_len)
        self.histogram.SetBitmap(icon.ConvertToScaledBitmap(bmp_size))

        btn_side_len = self.auto_clim.Size.height
        btn_size = wx.Size(btn_side_len, btn_side_len)
        self.histogram.SetMaxSize(btn_size)

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.visible, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.cmap, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.clims, 1, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.auto_clim, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.histogram, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.SetSizeHints(self)

        f = wx.BoxSizer(wx.VERTICAL)
        f.Add(sizer, 0, wx.EXPAND, 5)
        self._histogram_sizer = wx.BoxSizer(wx.HORIZONTAL)
        f.Add(self._histogram_sizer, 1, wx.EXPAND, 5)

        self.SetSizer(f)
        self.Layout()


class WxLutView(LutView):
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
        self._wxwidget.clims.SetValue(*clims)
        # FIXME: this is a hack.
        # it's required to make `set_auto_scale_without_signal` work as intended
        # But it's not a complete solution.  The general pattern of blocking signals
        # in Wx needs to be re-evaluated.
        wx.Yield()

    def set_channel_visible(self, visible: bool) -> None:
        self._wxwidget.visible.SetValue(visible)

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._wxwidget.Show()
        else:
            self._wxwidget.Hide()

    def close(self) -> None:
        self._wxwidget.Close()


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
            slider = WxLabeledSlider(self)
            slider.label.SetLabel(str(axis))
            slider.slider.Bind(wx.EVT_SLIDER, self._on_slider_changed)

            if isinstance(_coords, range):
                slider.setRange(_coords.start, _coords.stop - 1)
                slider.setSingleStep(_coords.step)
            else:
                slider.setRange(0, len(_coords) - 1)

            self.layout.Add(slider, 0, wx.EXPAND | wx.ALL, 5)
            self._sliders[axis] = slider
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
            choices=[ChannelMode.GRAYSCALE.value, ChannelMode.COMPOSITE.value],
            style=wx.CB_DROPDOWN,
        )

        # Reset zoom button
        self.reset_zoom_btn = wx.Button(self, label="Reset Zoom")

        # 3d view button
        self.ndims_btn = wx.ToggleButton(self, label="3D")

        # LUT layout (simple vertical grouping for LUT widgets)
        self.luts = wx.BoxSizer(wx.VERTICAL)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        btns.AddStretchSpacer()
        btns.Add(self.channel_mode_combo, 0, wx.ALL, 5)
        btns.Add(self.reset_zoom_btn, 0, wx.ALL, 5)
        btns.Add(self.ndims_btn, 0, wx.ALL, 5)

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
        inner.Add(btns, 0, wx.EXPAND)

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
        parent: wx.Window = None,
    ) -> None:
        self._data_model = data_model
        self._wxwidget = wdg = _WxArrayViewer(canvas_widget, parent)
        # Mapping of channel key to LutViews
        self._luts: dict[ChannelKey, WxLutView] = {}
        self._visible_axes: Sequence[AxisKey] = []

        # TODO: use emit_fast
        wdg.dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)
        wdg.channel_mode_combo.Bind(wx.EVT_COMBOBOX, self._on_channel_mode_changed)
        wdg.reset_zoom_btn.Bind(wx.EVT_BUTTON, self._on_reset_zoom_clicked)
        wdg.ndims_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_ndims_toggled)

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

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes  # no widget to control this yet

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._wxwidget.ndims_btn.SetValue(len(axes) == 3)

    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def add_lut_view(self, channel: ChannelKey) -> WxLutView:
        view = WxLutView(self.frontend_widget(), channel)
        self._luts[channel] = view

        def _on_histogram_requested(event: wx.CommandEvent) -> None:
            toggled = view._wxwidget.histogram.GetValue()
            if view._histogram:
                if toggled:
                    view._histogram.Show()
                else:
                    view._histogram.Hide()
            elif toggled:
                self.histogramRequested.emit(view._channel)

        self._wxwidget.luts.Add(view._wxwidget, 0, wx.EXPAND | wx.BOTTOM, 5)
        self._wxwidget.Layout()
        # TODO: This ugly
        view._wxwidget.histogram.Bind(wx.EVT_TOGGLEBUTTON, _on_histogram_requested)
        return view

    # TODO: Fix type
    def add_histogram(self, channel: ChannelKey, widget: Any) -> None:
        if lut := self._luts.get(channel, None):
            # FIXME: pygfx backend needs this to be widget._subwidget
            if hasattr(widget, "_subwidget"):
                widget = widget._subwidget

            if (parent := widget.GetParent()) and parent is not self:
                widget.Reparent(lut._wxwidget)  # Reparent widget to this frame
                if parent:
                    parent.Destroy()
                widget.Show()
            # FIXME: Yuck
            size = wx.Size(lut._wxwidget.Size)
            size.height += 100
            lut._wxwidget.SetSize(size)
            lut._wxwidget.SetMinSize(size)
            # widget.SetMinSize(wx.Size(100, 100))
            lut._wxwidget._histogram_sizer.Add(widget)
            lut._histogram = widget

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

    def set_progress_spinner_visible(self, visible: bool) -> None:
        if visible:
            self._wxwidget._progress_spinner.Show()
            self._wxwidget._top_info.Layout()
        else:
            self._wxwidget._progress_spinner.Hide()
