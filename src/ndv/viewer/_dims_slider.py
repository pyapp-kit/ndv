from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from warnings import warn

from qtpy.QtCore import QPoint, QPointF, QSize, Qt, Signal
from qtpy.QtGui import QCursor, QResizeEvent
from qtpy.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QElidingLabel, QLabeledRangeSlider
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

if TYPE_CHECKING:
    from typing import Hashable, Mapping, TypeAlias

    from PyQt6.QtGui import QResizeEvent

    # any hashable represent a single dimension in a AND array
    DimKey: TypeAlias = Hashable
    # any object that can be used to index a single dimension in an AND array
    Index: TypeAlias = int | slice
    # a mapping from dimension keys to indices (eg. {"x": 0, "y": slice(5, 10)})
    # this object is used frequently to query or set the currently displayed slice
    Indices: TypeAlias = Mapping[DimKey, Index]
    # mapping of dimension keys to the maximum value for that dimension
    Sizes: TypeAlias = Mapping[DimKey, int]


SS = """
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

QLabel { font-size: 12px; }

QRangeSlider { qproperty-barColor: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 80, 120, 0.2),
        stop:1 rgba(100, 80, 120, 0.4)
    )}

SliderLabel {
    font-size: 12px;
    color: white;
}
"""


class QtPopup(QDialog):
    """A generic popup window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(False)  # if False, then clicking anywhere else closes it
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)

        self.frame = QFrame(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.frame)
        layout.setContentsMargins(0, 0, 0, 0)

    def show_above_mouse(self, *args: Any) -> None:
        """Show popup dialog above the mouse cursor position."""
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() // 2, szhint.height() + 14)
        self.move(pos)
        self.resize(self.sizeHint())
        self.show()


class PlayButton(QPushButton):
    """Just a styled QPushButton that toggles between play and pause icons."""

    fpsChanged = Signal(float)

    PLAY_ICON = "bi:play-fill"
    PAUSE_ICON = "bi:pause-fill"

    def __init__(self, fps: float = 20, parent: QWidget | None = None) -> None:
        icn = QIconifyIcon(self.PLAY_ICON, color="#888888")
        icn.addKey(self.PAUSE_ICON, state=QIconifyIcon.State.On, color="#4580DD")
        super().__init__(icn, "", parent)
        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(0.5, 100)
        self.spin.setValue(fps)
        self.spin.valueChanged.connect(self.fpsChanged)
        self.setCheckable(True)
        self.setFixedSize(14, 18)
        self.setIconSize(QSize(16, 16))
        self.setStyleSheet("border: none; padding: 0; margin: 0;")

        self._popup = QtPopup(self)
        form = QFormLayout(self._popup.frame)
        form.setContentsMargins(6, 6, 6, 6)
        form.addRow("FPS", self.spin)

    def mousePressEvent(self, e: Any) -> None:
        if e and e.button() == Qt.MouseButton.RightButton:
            self._show_fps_dialog(e.globalPosition())
        else:
            super().mousePressEvent(e)

    def _show_fps_dialog(self, pos: QPointF) -> None:
        self._popup.show_above_mouse()


class LockButton(QPushButton):
    LOCK_ICON = "uis:unlock"
    UNLOCK_ICON = "uis:lock"

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        icn = QIconifyIcon(self.LOCK_ICON, color="#888888")
        icn.addKey(self.UNLOCK_ICON, state=QIconifyIcon.State.On, color="red")
        super().__init__(icn, text, parent)
        self.setCheckable(True)
        self.setFixedSize(20, 20)
        self.setIconSize(QSize(14, 14))
        self.setStyleSheet("border: none; padding: 0; margin: 0;")


class DimsSlider(QWidget):
    """A single slider in the DimsSliders widget.

    Provides a play/pause button that toggles animation of the slider value.
    Has a QLabeledSlider for the actual value.
    Adds a label for the maximum value (e.g. "3 / 10")
    """

    valueChanged = Signal(object, object)  # where object is int | slice

    def __init__(self, dimension_key: DimKey, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SS)
        self._slice_mode = False
        self._dim_key = dimension_key

        self._timer_id: int | None = None  # timer for play button
        self._play_btn = PlayButton(parent=self)
        self._play_btn.fpsChanged.connect(self.set_fps)
        self._play_btn.toggled.connect(self._toggle_animation)

        self._dim_key = dimension_key
        self._dim_label = QElidingLabel(str(dimension_key).upper())
        self._dim_label.setToolTip("Double-click to toggle slice mode")

        # note, this lock button only prevents the slider from updating programmatically
        # using self.setValue, it doesn't prevent the user from changing the value.
        self._lock_btn = LockButton(parent=self)

        self._pos_label = QSpinBox(self)
        self._pos_label.valueChanged.connect(self._on_pos_label_edited)
        self._pos_label.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._pos_label.setStyleSheet(
            "border: none; padding: 0; margin: 0; background: transparent"
        )
        self._out_of_label = QLabel(self)

        self._int_slider = QSlider(Qt.Orientation.Horizontal)
        self._int_slider.rangeChanged.connect(self._on_range_changed)
        self._int_slider.valueChanged.connect(self._on_int_value_changed)

        self._slice_slider = slc = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        slc.setHandleLabelPosition(QLabeledRangeSlider.LabelPosition.LabelsOnHandle)
        slc.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        slc.setVisible(False)
        slc.rangeChanged.connect(self._on_range_changed)
        slc.valueChanged.connect(self._on_slice_value_changed)

        self.installEventFilter(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._dim_label)
        layout.addWidget(self._int_slider)
        layout.addWidget(self._slice_slider)
        layout.addWidget(self._pos_label)
        layout.addWidget(self._out_of_label)
        layout.addWidget(self._lock_btn)
        self.setMinimumHeight(22)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        if isinstance(par := self.parent(), DimsSliders):
            par.resizeEvent(None)

    def mouseDoubleClickEvent(self, a0: Any) -> None:
        self._set_slice_mode(not self._slice_mode)
        super().mouseDoubleClickEvent(a0)

    def containMaximum(self, max_val: int) -> None:
        if max_val > self._int_slider.maximum():
            self._int_slider.setMaximum(max_val)
            if max_val > self._slice_slider.maximum():
                self._slice_slider.setMaximum(max_val)

    def setMaximum(self, max_val: int) -> None:
        self._int_slider.setMaximum(max_val)
        self._slice_slider.setMaximum(max_val)

    def setMinimum(self, min_val: int) -> None:
        self._int_slider.setMinimum(min_val)
        self._slice_slider.setMinimum(min_val)

    def containMinimum(self, min_val: int) -> None:
        if min_val < self._int_slider.minimum():
            self._int_slider.setMinimum(min_val)
            if min_val < self._slice_slider.minimum():
                self._slice_slider.setMinimum(min_val)

    def setRange(self, min_val: int, max_val: int) -> None:
        self._int_slider.setRange(min_val, max_val)
        self._slice_slider.setRange(min_val, max_val)

    def value(self) -> Index:
        if not self._slice_mode:
            return self._int_slider.value()  # type: ignore
        start, *_, stop = cast("tuple[int, ...]", self._slice_slider.value())
        if start == stop:
            return start
        return slice(start, stop)

    def setValue(self, val: Index) -> None:
        # variant of setValue that always updates the maximum
        self._set_slice_mode(isinstance(val, slice))
        if self._lock_btn.isChecked():
            return
        if isinstance(val, slice):
            start = int(val.start) if val.start is not None else 0
            stop = (
                int(val.stop) if val.stop is not None else self._slice_slider.maximum()
            )
            self._slice_slider.setValue((start, stop))
        else:
            self._int_slider.setValue(val)
            # self._slice_slider.setValue((val, val + 1))

    def forceValue(self, val: Index) -> None:
        """Set value and increase range if necessary."""
        if isinstance(val, slice):
            if isinstance(val.start, int):
                self.containMinimum(val.start)
            if isinstance(val.stop, int):
                self.containMaximum(val.stop)
        else:
            self.containMinimum(val)
            self.containMaximum(val)
        self.setValue(val)

    def _set_slice_mode(self, mode: bool = True) -> None:
        if mode == self._slice_mode:
            return
        self._slice_mode = bool(mode)
        self._slice_slider.setVisible(self._slice_mode)
        self._int_slider.setVisible(not self._slice_mode)
        # self._pos_label.setVisible(not self._slice_mode)
        self.valueChanged.emit(self._dim_key, self.value())

    def set_fps(self, fps: float) -> None:
        self._play_btn.spin.setValue(fps)
        self._toggle_animation(self._play_btn.isChecked())

    def _toggle_animation(self, checked: bool) -> None:
        if checked:
            if self._timer_id is not None:
                self.killTimer(self._timer_id)
            interval = int(1000 / self._play_btn.spin.value())
            self._timer_id = self.startTimer(interval)
        elif self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

    def timerEvent(self, event: Any) -> None:
        """Handle timer event for play button, move to the next frame."""
        # TODO
        # for now just increment the value by 1, but we should be able to
        # take FPS into account better and skip additional frames if the timerEvent
        # is delayed for some reason.
        inc = 1
        if self._slice_mode:
            val = cast(tuple[int, int], self._slice_slider.value())
            next_val = [v + inc for v in val]
            if next_val[1] > self._slice_slider.maximum():
                # wrap around, without going below the min handle
                next_val = [v - val[0] for v in val]
            self._slice_slider.setValue(next_val)
        else:
            ival = self._int_slider.value()
            ival = (ival + inc) % (self._int_slider.maximum() + 1)
            self._int_slider.setValue(ival)

    def _on_pos_label_edited(self) -> None:
        if self._slice_mode:
            self._slice_slider.setValue(
                (self._slice_slider.value()[0], self._pos_label.value())
            )
        else:
            self._int_slider.setValue(self._pos_label.value())

    def _on_range_changed(self, min: int, max: int) -> None:
        self._out_of_label.setText(f"| {max}")
        self._pos_label.setRange(min, max)
        self.resizeEvent(None)
        self.setVisible(min != max)

    def setVisible(self, visible: bool) -> None:
        if self._has_no_range():
            visible = False
        super().setVisible(visible)

    def _has_no_range(self) -> bool:
        if self._slice_mode:
            return bool(self._slice_slider.minimum() == self._slice_slider.maximum())
        return bool(self._int_slider.minimum() == self._int_slider.maximum())

    def _on_int_value_changed(self, value: int) -> None:
        self._pos_label.setValue(value)
        if not self._slice_mode:
            self.valueChanged.emit(self._dim_key, value)

    def _on_slice_value_changed(self, value: tuple[int, int]) -> None:
        self._pos_label.setValue(int(value[1]))
        with signals_blocked(self._int_slider):
            self._int_slider.setValue(int(value[0]))
        if self._slice_mode:
            self.valueChanged.emit(self._dim_key, slice(*value))


class DimsSliders(QWidget):
    """A Collection of DimsSlider widgets for each dimension in the data.

    Maintains the global current index and emits a signal when it changes.
    """

    valueChanged = Signal(dict)  # dict is of type Indices

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._locks_visible: bool | Mapping[DimKey, bool] = False
        self._sliders: dict[DimKey, DimsSlider] = {}
        self._current_index: dict[DimKey, Index] = {}
        self._invisible_dims: set[DimKey] = set()

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def __contains__(self, key: DimKey) -> bool:
        """Return True if the dimension key is present in the DimsSliders."""
        return key in self._sliders

    def slider(self, key: DimKey) -> DimsSlider:
        """Return the DimsSlider widget for the given dimension key."""
        return self._sliders[key]

    def value(self) -> Indices:
        """Return mapping of {dim_key -> current index} for each dimension."""
        return self._current_index.copy()

    def setValue(self, values: Indices) -> None:
        """Set the current index for each dimension.

        Parameters
        ----------
        values : Mapping[Hashable, int | slice]
            Mapping of {dim_key -> index} for each dimension.  If value is a slice,
            the slider will be in slice mode. If the dimension is not present in the
            DimsSliders, it will be added.
        """
        if self._current_index == values:
            return
        with signals_blocked(self):
            for dim, index in values.items():
                self.add_or_update_dimension(dim, index)
        # FIXME: i don't know why this this is ever empty ... only happens on pyside6
        if val := self.value():
            self.valueChanged.emit(val)

    def minima(self) -> Sizes:
        """Return mapping of {dim_key -> minimum value} for each dimension."""
        return {k: v._int_slider.minimum() for k, v in self._sliders.items()}

    def setMinima(self, values: Sizes) -> None:
        """Set the minimum value for each dimension.

        Parameters
        ----------
        values : Mapping[Hashable, int]
            Mapping of {dim_key -> minimum value} for each dimension.
        """
        for name, min_val in values.items():
            if name not in self._sliders:
                self.add_dimension(name)
            self._sliders[name].setMinimum(min_val)

    def maxima(self) -> Sizes:
        """Return mapping of {dim_key -> maximum value} for each dimension."""
        return {k: v._int_slider.maximum() for k, v in self._sliders.items()}

    def setMaxima(self, values: Sizes) -> None:
        """Set the maximum value for each dimension.

        Parameters
        ----------
        values : Mapping[Hashable, int]
            Mapping of {dim_key -> maximum value} for each dimension.
        """
        for name, max_val in values.items():
            if name not in self._sliders:
                self.add_dimension(name)
            self._sliders[name].setMaximum(max_val)

    def set_locks_visible(self, visible: bool | Mapping[DimKey, bool]) -> None:
        """Set the visibility of the lock buttons for all dimensions."""
        self._locks_visible = visible
        for dim, slider in self._sliders.items():
            viz = visible if isinstance(visible, bool) else visible.get(dim, False)
            slider._lock_btn.setVisible(viz)

    def add_dimension(self, key: DimKey, val: Index | None = None) -> None:
        """Add a new dimension to the DimsSliders widget.

        Parameters
        ----------
        key : Hashable
            The name of the dimension.
        val : int | slice, optional
            The initial value for the dimension. If a slice, the slider will be in
            slice mode.
        """
        self._sliders[key] = slider = DimsSlider(dimension_key=key, parent=self)
        if isinstance(self._locks_visible, dict) and key in self._locks_visible:
            slider._lock_btn.setVisible(self._locks_visible[key])
        else:
            slider._lock_btn.setVisible(bool(self._locks_visible))

        val_int = val.start if isinstance(val, slice) else val
        slider.setVisible(key not in self._invisible_dims)
        if isinstance(val_int, int):
            slider.setRange(val_int, val_int)
        elif isinstance(val_int, slice):
            slider.setRange(val_int.start or 0, val_int.stop or 1)

        val = val if val is not None else 0
        self._current_index[key] = val
        slider.forceValue(val)
        slider.valueChanged.connect(self._on_dim_slider_value_changed)
        cast("QVBoxLayout", self.layout()).addWidget(slider)

    def set_dimension_visible(self, key: DimKey, visible: bool) -> None:
        """Set the visibility of a dimension in the DimsSliders widget.

        Once a dimension is hidden, it will not be shown again until it is explicitly
        made visible again with this method.
        """
        if visible:
            self._invisible_dims.discard(key)
            if key in self._sliders:
                self._current_index[key] = self._sliders[key].value()
            else:
                self.add_dimension(key)
        else:
            self._invisible_dims.add(key)
            self._current_index.pop(key, None)
        if key in self._sliders:
            self._sliders[key].setVisible(visible)

    def remove_dimension(self, key: DimKey) -> None:
        """Remove a dimension from the DimsSliders widget."""
        try:
            slider = self._sliders.pop(key)
        except KeyError:
            warn(f"Dimension {key} not found in DimsSliders", stacklevel=2)
            return
        cast("QVBoxLayout", self.layout()).removeWidget(slider)
        slider.deleteLater()

    def _on_dim_slider_value_changed(self, key: DimKey, value: Index) -> None:
        self._current_index[key] = value
        self.valueChanged.emit(self.value())

    def add_or_update_dimension(self, key: DimKey, value: Index) -> None:
        """Add a dimension if it doesn't exist, otherwise update the value."""
        if key in self._sliders:
            self._sliders[key].forceValue(value)
        else:
            self.add_dimension(key, value)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        # align all labels
        if sliders := list(self._sliders.values()):
            for lbl in ("_dim_label", "_pos_label", "_out_of_label"):
                lbl_width = max(getattr(s, lbl).sizeHint().width() for s in sliders)
                for s in sliders:
                    getattr(s, lbl).setFixedWidth(lbl_width)

        super().resizeEvent(a0)

    def sizeHint(self) -> QSize:
        return super().sizeHint().boundedTo(QSize(9999, 0))
