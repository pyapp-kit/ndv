from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import wx
import wx.lib.newevent

from ndv._types import MouseMoveEvent
from ndv.models._array_display_model import ChannelMode

from ._wx_signal import WxSignal
from .range_slider import RangeSlider

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap

    AxisKey = int | None


class WxLUTWidget(wx.Panel):
    visibleChanged = WxSignal()
    autoscaleChanged = WxSignal()
    cmapChanged = WxSignal()
    climsChanged = WxSignal()
    gammaChanged = WxSignal()

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self._visible = wx.CheckBox(self, label="Visible")
        self._visible.SetValue(True)
        self._visible.Bind(wx.EVT_CHECKBOX, self._on_visible_changed)

        # Placeholder for the custom colormap combo box
        self._cmap = wx.ComboBox(
            self, choices=["gray", "green", "magenta"], style=wx.CB_DROPDOWN
        )
        self._cmap.Bind(wx.EVT_COMBOBOX, self._on_cmap_changed)

        # Placeholder for the QLabeledRangeSlider equivalent
        self._clims = RangeSlider(self, style=wx.SL_HORIZONTAL)
        self._clims.SetMax(65000)
        self._clims.SetValue(0, 65000)
        self._clims.Bind(wx.EVT_SLIDER, self._on_clims_changed)

        self._auto_clim = wx.ToggleButton(self, label="Auto")
        self._auto_clim.Bind(wx.EVT_TOGGLEBUTTON, self._on_autoscale_changed)

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self._visible, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self._cmap, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self._clims, 1, wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(self._auto_clim, 0, wx.ALIGN_CENTER_VERTICAL, 5)

        self.SetSizer(sizer)
        self.Layout()

    # Event Handlers
    def _on_visible_changed(self, event: wx.CommandEvent) -> None:
        self.visibleChanged.emit(self._visible.GetValue())

    def _on_cmap_changed(self, event: wx.CommandEvent) -> None:
        self.cmapChanged.emit(self._cmap.GetValue())

    def _on_clims_changed(self, event: wx.CommandEvent) -> None:
        self.climsChanged.emit(self._clims.GetValues())

    def _on_autoscale_changed(self, event: wx.CommandEvent) -> None:
        self.autoscaleChanged.emit(self._auto_clim.GetValue())

    # Public Methods
    def set_name(self, name: str) -> None:
        self._visible.SetLabel(name)

    def set_auto_scale(self, auto: bool) -> None:
        with self.autoscaleChanged.blocked():
            self._auto_clim.SetValue(auto)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        with self.cmapChanged.blocked():
            name = cmap.name.split(":")[-1]  # FIXME: this is a hack
            self._cmap.SetValue(name)

    def set_clims(self, clims: tuple[float, float]) -> None:
        with self.climsChanged.blocked():
            self._clims.SetValue(*clims)

    def set_lut_visible(self, visible: bool) -> None:
        with self.visibleChanged.blocked():
            self._visible.SetValue(visible)

    def setVisible(self, visible: bool) -> None:
        if visible:
            self.Show()
        else:
            self.Hide()


class WxLabeledSlider(wx.Panel):
    """A simple labeled slider widget for wxPython."""

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self.label = wx.StaticText(self)
        self.slider = wx.Slider(self, style=wx.HORIZONTAL)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def setRange(self, min_val: int, max_val: int) -> None:
        self.slider.SetMin(min_val)
        self.slider.SetMax(max_val)

    def setValue(self, value: int) -> None:
        self.slider.SetValue(value)

    def value(self) -> int:
        return self.slider.GetValue()  # type: ignore [no-any-return]

    def setSingleStep(self, step: int) -> None:
        self.slider.SetLineSize(step)


class WxDimsSliders(wx.Panel):
    currentIndexChanged = WxSignal()

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


class WxViewerView(wx.Frame):
    currentIndexChanged = WxSignal()
    resetZoomClicked = WxSignal()
    histogramRequested = WxSignal()
    mouseMoved = WxSignal()
    channelModeChanged = WxSignal()

    def __init__(self, canvas_widget: wx.Window, parent: wx.Window = None):
        super().__init__(parent)

        # patch the on_mouse_event from vispy _wx.CanvasBackend
        # to intercept various mouse events.
        if hasattr(canvas_widget, "on_mouse_event"):
            canvas_widget.Unbind(wx.EVT_MOUSE_EVENTS, canvas_widget)
            canvas_widget.Bind(wx.EVT_MOUSE_EVENTS, self._on_canvas_mouse_event)

        self._canvas = canvas_widget
        if canvas_widget.GetParent() is not self:
            parent = canvas_widget.GetParent()
            canvas_widget.Reparent(self)  # Reparent canvas_widget to this frame
            if parent:
                parent.Destroy()

        canvas_widget.Show()

        # Dynamic sliders for dimensions
        self._dims_sliders = WxDimsSliders(self)
        self._dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)

        # Labels for data and hover information
        self._data_info_label = wx.StaticText(self, label="")
        self._hover_info_label = wx.StaticText(self, label="")

        # Channel mode combo box
        self._channel_mode_combo = wx.ComboBox(
            self, choices=[x.value for x in ChannelMode], style=wx.CB_DROPDOWN
        )
        self._channel_mode_combo.Bind(wx.EVT_COMBOBOX, self._on_channel_mode_changed)

        # Reset zoom button
        self._reset_zoom_btn = wx.Button(self, label="Reset Zoom")
        self._reset_zoom_btn.Bind(wx.EVT_BUTTON, self._on_reset_zoom_clicked)

        # LUT layout (simple vertical grouping for LUT widgets)
        self._luts = wx.BoxSizer(wx.VERTICAL)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        btns.Add(self._channel_mode_combo, 0, wx.RIGHT, 5)
        btns.Add(self._reset_zoom_btn, 0, wx.RIGHT, 5)

        # Layout for the panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self._data_info_label, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self._canvas, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self._hover_info_label, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self._dims_sliders, 0, wx.EXPAND | wx.BOTTOM, 5)
        main_sizer.Add(self._luts, 0, wx.EXPAND, 5)
        main_sizer.Add(btns, 0, wx.EXPAND, 5)

        self.SetSizer(main_sizer)
        self.SetInitialSize(wx.Size(600, 800))
        self.Layout()

    def add_lut_view(self) -> WxLUTWidget:
        lut = WxLUTWidget(self)
        self._luts.Add(lut, 0, wx.EXPAND | wx.BOTTOM, 5)
        self.Layout()
        return lut

    def remove_lut_view(self, lut: WxLUTWidget) -> None:
        self._luts.Detach(lut)
        lut.Destroy()
        self.Layout()

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        self._dims_sliders.create_sliders(coords)
        self.Layout()

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        self._dims_sliders.hide_dimensions(axes_to_hide, show_remainder)
        self.Layout()

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        return self._dims_sliders.current_index()

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        self._dims_sliders.set_current_index(value)

    def set_data_info(self, text: str) -> None:
        self._data_info_label.SetLabel(text)

    def set_hover_info(self, text: str) -> None:
        self._hover_info_label.SetLabel(text)

    def set_channel_mode(self, mode: ChannelMode) -> None:
        self._channel_mode_combo.SetValue(mode)

    def _on_reset_zoom_clicked(self, event: wx.CommandEvent) -> None:
        self.resetZoomClicked.emit()

    def _on_channel_mode_changed(self, event: wx.CommandEvent) -> None:
        mode = self._channel_mode_combo.GetValue()
        self.channelModeChanged.emit(mode)

    def show(self) -> None:
        self.Show()

    def setVisible(self, visible: bool) -> None:
        if visible:
            self.Show()
        else:
            self.Hide()

    def _on_canvas_mouse_event(self, event: wx.MouseEvent) -> None:
        if event.Moving() or event.Dragging():  # mouse move event
            self.mouseMoved.emit(MouseMoveEvent(x=event.GetX(), y=event.GetY()))
        self._canvas.on_mouse_event(event)


if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, title="LUTWidget Test")
    lut_widget = WxLUTWidget(frame)

    lut_widget.climsChanged.connect(lambda clims: print(f"clims changed: {clims}"))

    frame.Show()
    app.MainLoop()
