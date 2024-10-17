from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Any

import ipywidgets as widgets
from psygnal import Signal

from .models._array_display_model import AxisKey
from .viewer._backends import get_canvas_class
from .viewer._backends._protocols import PImageHandle


class JupyterViewerView:
    currentIndexChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._canvas = get_canvas_class()()
        self._canvas.set_ndim(2)
        self._sliders: dict[Hashable, widgets.IntSlider] = {}
        self._slider_box = widgets.VBox([])
        self.layout = widgets.VBox([self._canvas.qwidget(), self._slider_box])

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
                continuous_update=False,
                orientation="horizontal",
            )
            sld.observe(self.on_slider_change, "value")
            sliders.append(sld)
            self._sliders[axis] = sld
        self._slider_box.children = sliders

        self.currentIndexChanged.emit()

    def on_slider_change(self, change: dict[str, Any]) -> None:
        """Emit signal when a slider value changes."""
        self.currentIndexChanged()

    def add_image_to_canvas(self, data: Any) -> PImageHandle:
        """Add image data to the canvas."""
        hdl = self._canvas.add_image(data)
        self._canvas.set_range()
        return hdl

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
