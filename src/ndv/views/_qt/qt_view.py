from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

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
from superqt import QCollapsible, QElidingLabel, QLabeledRangeSlider, QLabeledSlider
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv._types import AxisKey, MouseMoveEvent
from ndv.views.protocols import CursorType

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from ndv.views.protocols import PHistogramCanvas

SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 100, 100, 0.25),
        stop:1 rgba(100, 100, 100, 0.1)
    );
}

QLabel { font-size: 12px; }

QRangeSlider { qproperty-barColor: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 80, 120, 0.2),
        stop:1 rgba(100, 80, 120, 0.4)
    )}
"""


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
        self._cmap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._cmap.currentColormapChanged.connect(self.cmapChanged)
        for color in ["gray", "green", "magenta"]:
            self._cmap.addColormap(color)

        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)

        WHITE_SS = SLIDER_STYLE + "SliderLabel { font-size: 10px; color: white;}"
        self._clims.setStyleSheet(WHITE_SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**16)
        self._clims.valueChanged.connect(self.climsChanged)

        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setMaximumWidth(42)
        self._auto_clim.setCheckable(True)
        self._auto_clim.toggled.connect(self.autoscaleChanged)

        layout = QHBoxLayout(self)
        layout.setSpacing(5)
        # layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visible)
        layout.addWidget(self._cmap)
        layout.addWidget(self._clims)
        layout.addWidget(self._auto_clim)

    def set_name(self, name: str) -> None:
        self._visible.setText(name)

    # NOTE: it's important to block signals when setting values from the controller
    # to avoid loops, unnecessary updates, and unexpected behavior

    def set_auto_scale(self, auto: bool) -> None:
        with signals_blocked(self):
            self._auto_clim.setChecked(auto)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        with signals_blocked(self):
            self._cmap.setCurrentColormap(cmap)

    def set_clims(self, clims: tuple[float, float]) -> None:
        with signals_blocked(self):
            self._clims.setValue(clims)

    def set_lut_visible(self, visible: bool) -> None:
        with signals_blocked(self):
            self._visible.setChecked(visible)


class QDimsSliders(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}
        self.setStyleSheet(SLIDER_STYLE)

        layout = QFormLayout(self)
        layout.setSpacing(2)
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
        changed = False
        # only emit signal if the value actually changed
        # NOTE: this may be unnecessary, since usually the only thing calling
        # set_current_index is the controller, which already knows the value
        # however, we use this method directly in testing and it's nice to ensure.
        with signals_blocked(self):
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")
                if slider := self._sliders.get(axis):
                    if slider.value() != val:
                        changed = True
                        slider.setValue(val)
                else:  # pragma: no cover
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)
        if changed:
            self.currentIndexChanged.emit()


# this is a PView ... but that would make a metaclass conflict
class QtViewerView(QWidget):
    currentIndexChanged = Signal()
    resetZoomClicked = Signal()
    mouseMoved = Signal(MouseMoveEvent)

    def __init__(self, canvas_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        self._qcanvas = canvas_widget
        # TODO: this actually doesn't need to be in the QtViewerView at all
        # this could be patched at the level of the vispy/pygfx canvas
        # removing a need for the mouseMoved signal
        # Install an event filter so we can intercept mouse/key events
        self._qcanvas.installEventFilter(self)

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
        self._set_range_btn.clicked.connect(self.resetZoomClicked)

        self._btns = btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(0)
        btns.addStretch()
        btns.addWidget(self._channel_mode_btn)
        # btns.addWidget(self._ndims_btn)
        btns.addWidget(self._set_range_btn)
        # btns.addWidget(self._add_roi_btn)

        self._luts = QCollapsible(
            "LUTs",
            parent=self,
            expandedIcon=QIconifyIcon("bi:chevron-up", color="#888888"),
            collapsedIcon=QIconifyIcon("bi:chevron-down", color="#888888"),
        )

        # little hack to make the lut collapsible take up less space
        lut_layout = cast("QVBoxLayout", self._luts.layout())
        lut_layout.setContentsMargins(0, 1, 0, 1)
        lut_layout.setSpacing(0)
        if (
            # look-before-leap on private attribute that may change
            hasattr(self._luts, "_content")
            and (layout := self._luts._content.layout()) is not None
        ):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

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

    def set_data_info(self, text: str) -> None:
        """Set the data info text, above the canvas."""
        self._data_info_label.setText(text)

    def set_hover_info(self, text: str) -> None:
        """Set the hover info text, below the canvas."""
        self._hover_info_label.setText(text)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Event filter installed on the canvas to handle mouse events."""
        if event is None:
            return False  # pragma: no cover

        # here is where we get a chance to intercept mouse events before allowing the
        # canvas to respond to them.
        # Return `True` to prevent the event from being passed to the canvas.
        intercept = False
        # use children in case backend has a subwidget stealing events.
        if obj is self._qcanvas or obj in (self._qcanvas.children()):
            if isinstance(event, QMouseEvent):
                intercept |= self._canvas_mouse_event(event)
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(cast("QKeyEvent", event))
        return intercept

    def _canvas_mouse_event(self, ev: QMouseEvent) -> bool:
        # if ev.type() == QEvent.Type.MouseButtonPress:
        # ...
        if ev.type() == QEvent.Type.MouseMove:
            pos = ev.pos()
            self.mouseMoved.emit(MouseMoveEvent(x=pos.x(), y=pos.y()))
        # if ev.type() == QEvent.Type.MouseButtonRelease:
        #     ...
        return False


class QtHistogramView(QWidget):
    """A 'frontend' Qt wrapper around a 'backend' PHistogramView.

    Parameters
    ----------
    backend_widget: PHistogramView
        If a widget, set as the parent of this widget
    parent: QWidget | None
        If a widget, set as the parent of this widget
    """

    def __init__(self, backend_widget: PHistogramCanvas, parent: QWidget | None = None):
        super().__init__(parent)
        self._backend = backend_widget
        self._qwdg = cast(QWidget, self._backend.widget())
        self._qwdg.installEventFilter(self)
        self._pressed: bool = False

        # Log box
        self._log = QPushButton("Logarithmic")
        self._log.setToolTip("Toggle logarithmic (base-10) range scaling")
        self._log.setCheckable(True)
        self._log.toggled.connect(self._backend.set_range_log)

        # button to reset the zoom of the canvas
        # TODO: unify icons across all the view frontends in a new file
        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self._set_range_btn = QPushButton(set_range_icon, "", self)
        self._set_range_btn.setToolTip("Reset Pan/Zoom")
        self._set_range_btn.clicked.connect(self._resetZoom)

        self._btns = btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(0)
        btns.addStretch()
        btns.addWidget(self._log)
        btns.addWidget(self._set_range_btn)

        # Layout
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._qwdg)
        self._layout.addLayout(self._btns)

    def refresh(self) -> None:
        self._backend.refresh()

    # -- Private helpers -- #

    def _resetZoom(self) -> None:
        self._backend.set_domain(None)
        self._backend.set_range(None)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Event filter installed on the canvas to handle mouse events."""
        if event is None:
            return False  # pragma: no cover

        # here is where we get a chance to intercept mouse events before allowing the
        # canvas to respond to them.
        # Return `True` to prevent the event from being passed to the canvas.
        intercept = False
        # use children in case backend has a subwidget stealing events.
        if obj is self._qwdg or obj in (self._qwdg.children()):
            if isinstance(event, QMouseEvent):
                intercept |= self._canvas_mouse_event(event)
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(cast("QKeyEvent", event))
        return intercept

    def _canvas_mouse_event(self, ev: QMouseEvent) -> bool:
        pos = (ev.pos().x(), ev.pos().y())
        intercepted = False
        if ev.type() == QEvent.Type.MouseButtonPress:
            intercepted |= self._backend.on_mouse_press(pos)
            self._pressed = True
        if ev.type() == QEvent.Type.MouseMove:
            intercepted |= self._backend.on_mouse_move(pos)
            if not self._pressed:
                self._set_cursor(self._backend.get_cursor(pos))
        if ev.type() == QEvent.Type.MouseButtonRelease:
            intercepted |= self._backend.on_mouse_release(pos)
            self._pressed = False
        return intercepted

    def _set_cursor(self, type: CursorType) -> None:
        if type is CursorType.V_ARROW:
            self._qwdg.setCursor(Qt.CursorShape.SplitVCursor)
        elif type is CursorType.H_ARROW:
            self._qwdg.setCursor(Qt.CursorShape.SplitHCursor)
        elif type is CursorType.ALL_ARROW:
            self._qwdg.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self._qwdg.unsetCursor()
