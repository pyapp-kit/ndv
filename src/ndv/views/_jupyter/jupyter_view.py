from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import cmap
import ipywidgets as widgets
from psygnal import Signal

from ndv.models._array_display_model import ChannelMode

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from vispy.app.backends import _jupyter_rfb

    from ndv._types import AxisKey
    from ndv.views.protocols import PLutView, PSignal

# not entirely sure why it's necessary to specifically annotat signals as : PSignal
# i think it has to do with type variance?


class JupyterLutView:
    visibleChanged: PSignal = Signal(bool)
    autoscaleChanged: PSignal = Signal(bool)
    cmapChanged: PSignal = Signal(cmap.Colormap)
    climsChanged: PSignal = Signal(tuple)

    def __init__(self) -> None:
        # WIDGETS

        self._visible = widgets.Checkbox(value=True)
        self._cmap = widgets.Dropdown(
            options=["gray", "green", "magenta", "cubehelix"], value="gray"
        )
        self._clims = widgets.FloatRangeSlider(
            value=[0, 2**16],
            min=0,
            max=2**16,
            step=1,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
        )
        self._auto_clim = widgets.ToggleButton(
            value=True,
            description="Auto",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Auto scale",
            icon="check",
        )

        # LAYOUT

        self.layout = widgets.HBox(
            [
                self._visible,
                self._cmap,
                self._clims,
                self._auto_clim,
            ]
        )

        # CONNECTIONS

        self._visible.observe(self._on_visible_changed, names="value")
        self._cmap.observe(self._on_cmap_changed, names="value")
        self._clims.observe(self._on_clims_changed, names="value")
        self._auto_clim.observe(self._on_autoscale_changed, names="value")

    # ------------------ emit changes to the controller ------------------

    def _on_clims_changed(self, change: dict[str, Any]) -> None:
        self.climsChanged.emit(self._clims.value)

    def _on_visible_changed(self, change: dict[str, Any]) -> None:
        self.visibleChanged.emit(self._visible.value)

    def _on_cmap_changed(self, change: dict[str, Any]) -> None:
        self.cmapChanged.emit(cmap.Colormap(self._cmap.value))

    def _on_autoscale_changed(self, change: dict[str, Any]) -> None:
        self.autoscaleChanged.emit(self._auto_clim.value)

    # ------------------ receive changes from the controller ---------------

    def set_name(self, name: str) -> None:
        self._visible.description = name

    # NOTE: it's important to block signals when setting values from the controller
    # to avoid loops, unnecessary updates, and unexpected behavior

    def set_auto_scale(self, auto: bool) -> None:
        with self.autoscaleChanged.blocked():  # type: ignore [attr-defined]
            self._auto_clim.value = auto

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        with self.cmapChanged.blocked():  # type: ignore [attr-defined]
            self._cmap.value = cmap.name.split(":")[-1]  # FIXME: this is a hack

    def set_clims(self, clims: tuple[float, float]) -> None:
        with self.climsChanged.blocked():  # type: ignore [attr-defined]
            self._clims.value = clims

    def set_lut_visible(self, visible: bool) -> None:
        with self.visibleChanged.blocked():  # type: ignore [attr-defined]
            self._visible.value = visible

    def setVisible(self, visible: bool) -> None:
        # show or hide the actual widget itself
        self.layout.layout.display = "flex" if visible else "none"


# this is a PView
class JupyterViewerView:
    currentIndexChanged: PSignal = Signal()
    resetZoomClicked: PSignal = Signal()
    channelModeChanged: PSignal = Signal(ChannelMode)

    def __init__(
        self, canvas_widget: _jupyter_rfb.CanvasBackend, **kwargs: Any
    ) -> None:
        super().__init__()

        # WIDGETS
        self._canvas_widget = canvas_widget

        self._sliders: dict[Hashable, widgets.IntSlider] = {}
        self._slider_box = widgets.VBox([])
        self._data_info_label = widgets.Label()
        self._hover_info_label = widgets.Label()

        # the button that controls the display mode of the channels
        self._channel_mode_combo = widgets.Dropdown(
            options=[x.value for x in ChannelMode], value=str(ChannelMode.GRAYSCALE)
        )

        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

        # LAYOUT

        self.layout = widgets.VBox(
            [
                self._data_info_label,
                self._canvas_widget,
                self._hover_info_label,
                self._slider_box,
                self._channel_mode_combo,
            ]
        )

        # CONNECTIONS

        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
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
        with self.currentIndexChanged.blocked():  # type: ignore [attr-defined]
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

    def add_lut_view(self) -> JupyterLutView:
        """Add a LUT view to the viewer."""
        wdg = JupyterLutView()
        self.layout.children = (*self.layout.children, wdg.layout)

        # this cast is necessary because psygnal.Signal() is not being recognized
        # as a PSignalDescriptor by the type checker
        return wdg

    def remove_lut_view(self, view: PLutView) -> None:
        """Remove a LUT view from the viewer."""
        _view = cast("JupyterLutView", view)
        self.layout.children = tuple(
            wdg for wdg in self.layout.children if wdg != _view.layout
        )

    def show(self) -> None:
        """Show the viewer."""
        from IPython.display import display

        display(self.layout)  # type: ignore [no-untyped-call]

    def hide(self) -> None:
        """Hide the viewer."""
        from IPython.display import clear_output

        clear_output()  # type: ignore [no-untyped-call]

    def set_data_info(self, data_info: str) -> None:
        self._data_info_label.value = data_info

    def set_hover_info(self, hover_info: str) -> None:
        self._hover_info_label.value = hover_info

    def set_channel_mode(self, mode: ChannelMode) -> None:
        with self.channelModeChanged.blocked():  # type: ignore [attr-defined]
            self._channel_mode_combo.value = mode.value

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        """Emit signal when a slider value changes."""
        self.currentIndexChanged.emit()

    def _on_channel_mode_changed(self, change: dict[str, Any]) -> None:
        """Emit signal when the channel mode changes."""
        self.channelModeChanged.emit(ChannelMode(change["new"]))
