from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import wx
import wx.lib.newevent
from psygnal import Signal

from ndv.models._array_display_model import ChannelMode
from ndv.views._wx._labeled_slider import WxLabeledSlider
from ndv.views.bases import ArrayView, LutView

from .range_slider import RangeSlider

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap

    from ndv._types import AxisKey


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

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.visible, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.cmap, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.clims, 1, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self.auto_clim, 0, wx.ALIGN_CENTER_VERTICAL, 5)

        self.SetSizer(sizer)
        self.Layout()


class WxLutView(LutView):
    def __init__(self, parent: wx.Window) -> None:
        super().__init__()
        self._wxwidget = wdg = _WxLUTWidget(parent)
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
            self._model.clims = self._wxwidget.clims.GetValues()
            self._model.autoscale = False

    def _on_autoscale_changed(self, event: wx.CommandEvent) -> None:
        if self._model:
            self._model.autoscale = self._wxwidget.auto_clim.GetValue()

    # Public Methods
    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def set_channel_name(self, name: str) -> None:
        self._wxwidget.visible.SetLabel(name)

    def set_auto_scale(self, auto: bool) -> None:
        self._wxwidget.auto_clim.SetValue(auto)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        name = cmap.name.split(":")[-1]  # FIXME: this is a hack
        self._wxwidget.cmap.SetValue(name)

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._wxwidget.clims.SetValue(*clims)

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

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
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

        # Channel mode combo box
        self.channel_mode_combo = wx.ComboBox(
            self, choices=[x.value for x in ChannelMode], style=wx.CB_DROPDOWN
        )

        # Reset zoom button
        self.reset_zoom_btn = wx.Button(self, label="Reset Zoom")

        # LUT layout (simple vertical grouping for LUT widgets)
        self.luts = wx.BoxSizer(wx.VERTICAL)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        btns.Add(self.channel_mode_combo, 0, wx.RIGHT, 5)
        btns.Add(self.reset_zoom_btn, 0, wx.RIGHT, 5)

        # Layout for the panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self._data_info_label, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self._canvas, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self._hover_info_label, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self.dims_sliders, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self.luts, 0, wx.EXPAND, 5)
        main_sizer.Add(btns, 0, wx.EXPAND, 5)

        self.SetSizer(main_sizer)
        self.SetInitialSize(wx.Size(600, 800))
        self.Layout()


class WxArrayView(ArrayView):
    def __init__(self, canvas_widget: wx.Window, parent: wx.Window = None) -> None:
        self._wxwidget = wdg = _WxArrayViewer(canvas_widget, parent)

        # TODO: use emit_fast
        wdg.dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)
        wdg.channel_mode_combo.Bind(wx.EVT_COMBOBOX, self._on_channel_mode_changed)
        wdg.reset_zoom_btn.Bind(wx.EVT_BUTTON, self._on_reset_zoom_clicked)

    def _on_channel_mode_changed(self, event: wx.CommandEvent) -> None:
        mode = self._wxwidget.channel_mode_combo.GetValue()
        self.channelModeChanged.emit(mode)

    def _on_reset_zoom_clicked(self, event: wx.CommandEvent) -> None:
        self.resetZoomClicked.emit()

    def frontend_widget(self) -> wx.Window:
        return self._wxwidget

    def add_lut_view(self) -> WxLutView:
        view = WxLutView(self.frontend_widget())
        self._wxwidget.luts.Add(view._wxwidget, 0, wx.EXPAND | wx.BOTTOM, 5)
        self._wxwidget.Layout()
        return view

    def remove_lut_view(self, lut: LutView) -> None:
        wxwdg = cast("_WxLUTWidget", lut.frontend_widget())
        self._wxwidget.luts.Detach(wxwdg)
        wxwdg.Destroy()
        self._wxwidget.Layout()

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
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
