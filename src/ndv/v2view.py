from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from superqt import QLabeledSlider

from .models._array_display_model import AxisKey
from .viewer._backends import get_canvas_class
from .viewer._backends._protocols import PImageHandle


class ViewerView(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}
        self._canvas = get_canvas_class()()
        self._canvas.set_ndim(2)
        layout = QVBoxLayout(self)
        self._slider_layout = QFormLayout()
        self._slider_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout.addWidget(self._canvas.qwidget())
        layout.addLayout(self._slider_layout)

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        for axis, _coords in coords.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.valueChanged.connect(self.currentIndexChanged.emit)
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            self._slider_layout.addRow(str(axis), sld)
            self._sliders[axis] = sld
        self.currentIndexChanged.emit()

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
                self._slider_layout.setRowVisible(slider, False)
            elif show_remainder:
                self._slider_layout.setRowVisible(slider, True)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        for axis, val in value.items():
            if isinstance(val, slice):
                raise NotImplementedError("Slices are not supported yet")
            self._sliders[axis].setValue(val)
