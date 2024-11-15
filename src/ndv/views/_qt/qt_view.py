from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Any, cast

import cmap
from qtpy.QtCore import QEvent, QObject, Qt, Signal
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QLabeledRangeSlider, QLabeledSlider
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv._types import AxisKey
from ndv.views import get_canvas_class
from ndv.views._qt._dims_slider import SS
from ndv.views.protocols import CanvasElement, PImageHandle, PRoiHandle


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


class ROIButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setToolTip("Add ROI")
        self.setIcon(QIconifyIcon("mdi:vector-rectangle"))


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


class QViewerView(QWidget):
    currentIndexChanged = Signal()
    boundingBoxChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # NOTE: it's conceivable we'll want the controller itself to handle the canvas
        self._canvas = get_canvas_class()()
        self._canvas.set_ndim(2)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        self._dims_sliders = QDimsSliders(self)
        self._dims_sliders.currentIndexChanged.connect(self.currentIndexChanged)

        self._channel_mode_btn = QPushButton("Channel")
        self._roi_handle: PRoiHandle | None = None
        self._selection: CanvasElement | None = None

        # button to draw ROIs
        self._add_roi_btn = ROIButton()
        self._add_roi_btn.toggled.connect(self._on_add_roi_clicked)

        self._btns = btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(0)
        btns.addStretch()
        btns.addWidget(self._channel_mode_btn)
        # btns.addWidget(self._ndims_btn)
        # btns.addWidget(self._set_range_btn)
        btns.addWidget(self._add_roi_btn)

        self._luts = QCollapsible()
        self._luts.layout().setSpacing(0)
        self._luts.setCollapsedIcon(QIconifyIcon("bi:chevron-down", color="#888888"))
        self._luts.setExpandedIcon(QIconifyIcon("bi:chevron-up", color="#888888"))
        qwidget = self._canvas.qwidget()
        qwidget.installEventFilter(self)
        layout.addWidget(qwidget, 1)
        layout.addWidget(self._dims_sliders)
        layout.addWidget(self._luts)
        layout.addLayout(btns)

    def add_lut_view(self) -> QLUTWidget:
        wdg = QLUTWidget(self)
        self._luts.addWidget(wdg)
        return wdg

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

    def refresh(self) -> None:
        """Refresh the view."""
        self._canvas.refresh()

    def set_visible_axes(self, axes: Sequence[Hashable]) -> None:
        """Set the visible axes."""
        self._visible_axes.setText(", ".join(map(str, axes)))

    def setBoundingBox(self, min: Sequence[int], max: Sequence[int]) -> None:
        if self._roi_handle is None:
            self._roi_handle = self._canvas.add_roi()

        # Only 2D vertices currently supported
        if len(min) != 2 or len(max) != 2:
            raise ValueError("Only 2D Boxes currently supported")
        self._roi_handle.vertices = [
            (min[0], min[1]),
            (max[0], min[1]),
            (max[0], max[1]),
            (min[0], max[1]),
        ]

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Event filter installed on the canvas to handle mouse events."""
        if event is None:
            return False  # pragma: no cover

        # here is where we get a chance to intercept mouse events before passing them
        # to the canvas. Return `True` to prevent the event from being passed to
        # the backend widget.
        intercept = False
        # use children in case backend has a subwidget stealing events.
        qcanvas = self._canvas.qwidget()
        if obj is qcanvas or obj in (qcanvas.children()):
            if isinstance(event, QMouseEvent):
                intercept |= self._canvas_mouse_event(event)
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(cast("QKeyEvent", event))
        return intercept

    def _canvas_mouse_event(self, ev: QMouseEvent) -> bool:
        intercept = False
        if ev.type() == QEvent.Type.MouseButtonPress:
            if self._add_roi_btn.isChecked():
                intercept |= self._begin_roi(ev)
            intercept |= self._grab_roi(ev)
            return intercept
        if ev.type() == QEvent.Type.MouseMove:
            intercept = self._move_roi(ev)
            intercept |= self._update_cursor(ev)
            return intercept
        if ev.type() == QEvent.Type.MouseButtonRelease:
            intercept |= self._deselect_roi_btn()
            return intercept
        return intercept

    # FIXME: This is ugly
    def _begin_roi(self, event: QMouseEvent) -> bool:
        self._roi_handle = self._canvas.add_roi()
        if self._roi_handle:
            ev_pos = event.position()
            pos = self._canvas.canvas_to_world((ev_pos.x(), ev_pos.y()))
            self._roi_handle.move(pos)
            self._roi_handle.visible = True
        return False

    def _grab_roi(self, ev: QMouseEvent) -> bool:
        ev_pos = (ev.position().x(), ev.position().y())
        pos = self._canvas.canvas_to_world(ev_pos)
        # TODO why does the canvas need this point untransformed??
        elements = self._canvas.elements_at(ev_pos)
        # Deselect prior selection before editing new selection
        if self._selection:
            self._selection.selected = False
        for e in elements:
            if e.can_select:
                e.start_move(pos)
                # Select new selection
                self._selection = e
                self._selection.selected = True
                return False
        return False

    def _move_roi(self, event: QMouseEvent) -> bool:
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self._selection and self._selection.selected:
                ev_pos = event.pos()
                pos = self._canvas.canvas_to_world((ev_pos.x(), ev_pos.y()))
                self._selection.move(pos)
                # If we are moving the object, we don't want to move the camera
                return True
        return False

    def _update_cursor(self, event: QMouseEvent) -> bool:
        # Avoid changing the cursor when dragging
        if event.buttons() != Qt.MouseButton.NoButton:
            return False
        qcanvas = self._canvas.qwidget()
        # When "creating" a ROI, use CrossCursor
        if self._add_roi_btn.isChecked():
            qcanvas.setCursor(Qt.CursorShape.CrossCursor)
            return False
        # If any local elements have a preference, use it
        pos = (event.pos().x(), event.pos().y())
        for e in self._canvas.elements_at(pos):
            if (pref := e.cursor_at(pos)) is not None:
                qcanvas.setCursor(pref)
                return False
        # Otherwise, normal cursor
        qcanvas.setCursor(Qt.CursorShape.ArrowCursor)
        return False

    def _set_roi(
        self,
        vertices: list[tuple[float, float]] | None = None,
        color: Any = None,
        border_color: Any = None,
    ) -> None:
        """Set the properties of the ROI overlaid on the displayed data.

        Properties
        ----------
        vertices : list[tuple[float, float]] | None
            The vertices of the ROI.
        color : str, tuple, list, array, Color, or int
            The fill color.  Can be any "ColorLike".
        border_color : str, tuple, list, array, Color, or int
            The border color.  Can be any "ColorLike".
        """
        # Remove the old ROI
        if self._roi_handle:
            self._roi_handle.remove()

        self._roi_handle = self._canvas.add_roi(
            vertices=vertices, color=color, border_color=border_color
        )

    def _on_add_roi_clicked(self, checked: bool) -> None:
        if checked:
            # Add new roi
            self._set_roi()

    def _deselect_roi_btn(self) -> bool:
        if self._add_roi_btn.isChecked():
            self._add_roi_btn.click()
        # TODO: Improve code
        if self._roi_handle and self._selection is not None:
            self.boundingBoxChanged.emit()
        return False
