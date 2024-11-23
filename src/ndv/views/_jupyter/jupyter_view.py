from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cmap
import ipywidgets as widgets
from psygnal import Signal

from ndv._types import MouseMoveEvent

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from ndv._types import AxisKey
    from ndv.views.protocols import PLutView, PSignal


class JupyterLutView:
    visibleChanged = Signal(bool)
    autoscaleChanged = Signal(bool)
    cmapChanged = Signal(cmap.Colormap)
    climsChanged = Signal(tuple)

    def __init__(self) -> None:
        self._visible = widgets.Checkbox(value=True)
        self._visible.observe(self._on_visible_changed, names="value")

        self._cmap = widgets.Dropdown(
            options=["gray", "green", "magenta"], value="gray"
        )
        self._cmap.observe(self._on_cmap_changed, names="value")

        self._clims = widgets.FloatRangeSlider(
            value=[0, 2**16],
            min=0,
            max=2**16,
            step=1,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
        )
        self._clims.observe(self._on_clims_changed, names="value")

        self._auto_clim = widgets.ToggleButton(
            value=True,
            description="Auto",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Auto scale",
            icon="check",
        )
        self._auto_clim.observe(self._on_autoscale_changed, names="value")

        self.layout = widgets.HBox(
            [self._visible, self._cmap, self._clims, self._auto_clim]
        )

    def _on_clims_changed(self, change: dict[str, Any]) -> None:
        self.climsChanged(self._clims.value)

    def _on_visible_changed(self, change: dict[str, Any]) -> None:
        self.visibleChanged(self._visible.value)

    def _on_cmap_changed(self, change: dict[str, Any]) -> None:
        self.cmapChanged(cmap.Colormap(self._cmap.value))

    def _on_autoscale_changed(self, change: dict[str, Any]) -> None:
        self.autoscaleChanged(self._auto_clim.value)

    def setName(self, name: str) -> None:
        self._visible.description = name

    def setAutoScale(self, auto: bool) -> None:
        self._auto_clim.value = auto

    def setColormap(self, cmap: cmap.Colormap) -> None:
        self._cmap.value = cmap.name

    def setClims(self, clims: tuple[float, float]) -> None:
        self._clims.value = clims

    def setLutVisible(self, visible: bool) -> None:
        self._visible.value = visible


# this is a PView
class JupyterViewerView:
    # not sure why this annotation is necessary ... something wrong with PSignal
    currentIndexChanged: PSignal = Signal()
    resetZoomClicked: PSignal = Signal()
    mouseMoved: PSignal = Signal(MouseMoveEvent)

    def __init__(self, canvas_widget: Any, **kwargs: Any) -> None:
        super().__init__()
        self._canvas_widget = canvas_widget
        self._sliders: dict[Hashable, widgets.IntSlider] = {}
        self._slider_box = widgets.VBox([])
        # `qwidget` is obviously a misnomer here.  it works, because vispy is smart
        # enough to return a widget that ipywidgets can display in the appropriate
        # context, but we should be managing that more explicitly ourselves.
        self.layout = widgets.VBox([self._canvas_widget, self._slider_box])

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
            sld.observe(self.on_slider_change, "value")
            sliders.append(sld)
            self._sliders[axis] = sld
        self._slider_box.children = sliders

        self.currentIndexChanged.emit()

    def on_slider_change(self, change: dict[str, Any]) -> None:
        """Emit signal when a slider value changes."""
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
        for axis, val in value.items():
            if isinstance(val, slice):
                raise NotImplementedError("Slices are not supported yet")
            self._sliders[axis].value = val

    def add_lut_view(self) -> PLutView:
        """Add a LUT view to the viewer."""
        wdg = JupyterLutView()
        self.layout.children = (*self.layout.children, wdg.layout)
        return wdg

    def show(self) -> None:
        """Show the viewer."""
        from IPython.display import display

        display(self.layout)  # type: ignore [no-untyped-call]

    def set_data_info(self, data_info: str) -> None: ...
    def set_hover_info(self, hover_info: str) -> None: ...
