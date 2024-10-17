from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Any, cast

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from superqt import QLabeledSlider

from .models._array_display_model import AxisKey
from .viewer._backends import get_canvas_class
from .viewer._backends._protocols import PImageHandle


class QDimsSliders(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        layout = cast("QFormLayout", self.layout())
        for axis, _coords in coords.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.valueChanged.connect(self.currentIndexChanged)
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            layout.addRow(str(axis), sld)
            self._sliders[axis] = sld
        self.currentIndexChanged.emit()

    def hide_dimensions(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        layout = cast("QFormLayout", self.layout())
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                layout.setRowVisible(slider, False)
            elif show_remainder:
                layout.setRowVisible(slider, True)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        for axis, val in value.items():
            if isinstance(val, slice):
                raise NotImplementedError("Slices are not supported yet")
            self._sliders[axis].setValue(val)


class QViewerView(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._canvas = get_canvas_class()()
        self._canvas.set_ndim(2)
        layout = QVBoxLayout(self)
        self._dims_sliders = QDimsSliders(self)
        self._dims_sliders.currentIndexChanged.connect(self.currentIndexChanged)
        layout.addWidget(self._canvas.qwidget())
        layout.addWidget(self._dims_sliders)

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        self._dims_sliders.create_sliders(coords)

    def add_image_to_canvas(self, data: Any) -> PImageHandle:
        """Add image data to the canvas."""
        hdl = self._canvas.add_image(data)
        self._canvas.set_range()
        return hdl

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        """Hide sliders based on visible axes."""
        self._dims_sliders.hide_dimensions(axes_to_hide, show_remainder)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return self._dims_sliders.current_index()

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        self._dims_sliders.set_current_index(value)
