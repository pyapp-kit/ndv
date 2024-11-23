from collections.abc import Container, Hashable, Mapping, Sequence
from typing import cast

import cmap
import numpy as np
from qtpy.QtCore import QEvent, QObject, Qt, Signal
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QElidingLabel, QLabeledRangeSlider, QLabeledSlider
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv._types import AxisKey
from ndv.views._qt._dims_slider import SS


class CmapCombo(QColormapComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, allow_user_colormaps=True, add_colormap_text="Add...")
        self.setMinimumSize(120, 21)
        # self.setStyleSheet("background-color: transparent;")

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.findChild(QFrame)
        popup.setMinimumWidth(self.width() + 100)
        popup.move(popup.x(), popup.y() - self.height() - popup.height())


class QLUTWidget(QWidget):
    visibleChanged = Signal(bool)
    autoscaleChanged = Signal(bool)
    cmapChanged = Signal(cmap.Colormap)
    climsChanged = Signal(tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._visible = QCheckBox()
        self._visible.setChecked(True)
        self._visible.toggled.connect(self.visibleChanged)

        self._cmap = CmapCombo()
        self._cmap.currentColormapChanged.connect(self.cmapChanged)
        for color in ["gray", "green", "magenta"]:
            self._cmap.addColormap(color)

        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**16)
        self._clims.valueChanged.connect(self.climsChanged)

        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setMaximumWidth(42)
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self.autoscaleChanged)

        layout = QHBoxLayout(self)
        # layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visible)
        layout.addWidget(self._cmap)
        layout.addWidget(self._clims)
        layout.addWidget(self._auto_clim)

    def setName(self, name: str) -> None:
        self._visible.setText(name)

    def setAutoScale(self, auto: bool) -> None:
        with signals_blocked(self):
            self._auto_clim.setChecked(auto)

    def setColormap(self, cmap: cmap.Colormap) -> None:
        with signals_blocked(self):
            self._cmap.setCurrentColormap(cmap)

    def setClims(self, clims: tuple[float, float]) -> None:
        with signals_blocked(self):
            self._clims.setValue(clims)

    def setLutVisible(self, visible: bool) -> None:
        with signals_blocked(self):
            self._visible.setChecked(visible)


class QDimsSliders(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        layout = cast("QFormLayout", self.layout())
        for axis, _coords in coords.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.valueChanged.connect(self.currentIndexChanged)
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            else:
                sld.setRange(0, len(_coords) - 1)
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


# this is a PView ... but that would make a metaclass conflict
class QViewerView(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, canvas_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self._qcanvas = canvas_widget

        self._dims_sliders = QDimsSliders(self)
        self._dims_sliders.currentIndexChanged.connect(self.currentIndexChanged)

        # place to display dataset summary
        self._data_info_label = QElidingLabel("", parent=self)
        # place to display arbitrary text
        self._hover_info_label = QElidingLabel("", self)

        # the button that controls the display mode of the channels
        self._channel_mode_btn = QPushButton("Channel")
        # button to reset the zoom of the canvas
        # TODO: unify icons across all the view frontends in a new file
        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self._set_range_btn = QPushButton(set_range_icon, "", self)
        self._set_range_btn.clicked.connect(self._reset_zoom)

        self._btns = btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(0)
        btns.addStretch()
        btns.addWidget(self._channel_mode_btn)
        # btns.addWidget(self._ndims_btn)
        btns.addWidget(self._set_range_btn)
        # btns.addWidget(self._add_roi_btn)

        self._luts = QCollapsible()
        cast("QLayout", self._luts.layout()).setSpacing(0)
        self._luts.setCollapsedIcon(QIconifyIcon("bi:chevron-down", color="#888888"))
        self._luts.setExpandedIcon(QIconifyIcon("bi:chevron-up", color="#888888"))

        # above the canvas
        info_widget = QWidget()
        info = QHBoxLayout(info_widget)
        info.setContentsMargins(0, 0, 0, 2)
        info.setSpacing(0)
        info.addWidget(self._data_info_label)
        info_widget.setFixedHeight(16)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(info_widget)
        layout.addWidget(self._qcanvas, 1)
        layout.addWidget(self._hover_info_label)
        layout.addWidget(self._dims_sliders)
        layout.addWidget(self._luts)
        layout.addLayout(btns)

    def add_lut_view(self) -> QLUTWidget:
        wdg = QLUTWidget(self)
        self._luts.addWidget(wdg)
        return wdg

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        self._dims_sliders.create_sliders(coords)

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

    def refresh(self) -> None:
        """Refresh the view."""
        self._canvas.refresh()

    def set_visible_axes(self, axes: Sequence[Hashable]) -> None:
        """Set the visible axes."""
        self._visible_axes.setText(", ".join(map(str, axes)))

    def set_data_info(self, text: str) -> None:
        """Set the data info text, above the canvas."""
        self._data_info_label.setText(text)

    def set_hover_info(self, text: str) -> None:
        """Set the hover info text, below the canvas."""
        self._hover_info_label.setText(text)

    # FIXME:
    # this method is called when we click the reset zoom button
    # however, the controller currently knows nothing about it, because this view
    # itself directly controls the canvas.  We probably need the controller to control
    # the canvas.
    def _reset_zoom(self) -> None:
        self._canvas.set_range()

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Event filter installed on the canvas to handle mouse events."""
        if event is None:
            return False  # pragma: no cover

        # here is where we get a chance to intercept mouse events before passing them
        # to the canvas. Return `True` to prevent the event from being passed to
        # the backend widget.
        intercept = False
        # use children in case backend has a subwidget stealing events.
        if obj is self._qcanvas or obj in (self._qcanvas.children()):
            if isinstance(event, QMouseEvent):
                intercept |= self._canvas_mouse_event(event)
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(cast("QKeyEvent", event))
        return intercept

    def _canvas_mouse_event(self, ev: QMouseEvent) -> bool:
        intercept = False
        # if ev.type() == QEvent.Type.MouseButtonPress:
        # ...
        if ev.type() == QEvent.Type.MouseMove:
            intercept = self._update_hover_info(ev)
            return intercept
        # if ev.type() == QEvent.Type.MouseButtonRelease:
        #     ...
        return False

    def _update_hover_info(self, event: QMouseEvent) -> bool:
        """Update text of hover_info_label with data value(s) at point."""
        point = event.pos()
        x, y, _z = self._canvas.canvas_to_world((point.x(), point.y()))
        # TODO: handle 3D data
        if (x < 0 or y < 0) or self._ndims == 3:  # pragma: no cover
            self._hover_info_label.setText("")
            return False

        x = int(x)
        y = int(y)
        text = f"[{y}, {x}]"
        for n, handles in enumerate(self._img_handles.values()):
            channels = []
            for handle in handles:
                try:
                    # here, we're retrieving the value from the in-memory data
                    # stored by the backend visual, rather than querying the data itself
                    # this is a quick workaround to get the value without having to
                    # worry about higher dimensions in the data source (since the
                    # texture has already been reduced to 2D). But a more complete
                    # implementation would gather the full current nD index and query
                    # the data source directly.
                    value = handle.data[y, x]
                    if isinstance(value, (np.floating, float)):
                        value = f"{value:.2f}"
                    channels.append(f" {n}: {value}")
                except IndexError:
                    # we're out of bounds
                    # if we eventually have multiple image sources with different
                    # extents, this will need to be handled.  here, we just skip
                    self._hover_info_label.setText("")
                    return False
                break  # only getting one handle per channel
            text += ",".join(channels)
        self._hover_info_label.setText(text)
        return False
