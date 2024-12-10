from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import cmap
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import (
    QCollapsible,
    QElidingLabel,
    QEnumComboBox,
    QLabeledRangeSlider,
    QLabeledSlider,
)
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv.models._array_display_model import ChannelMode

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from qtpy.QtGui import QIcon

    from ndv._types import AxisKey

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
        layout.setContentsMargins(0, 0, 0, 0)
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

    def set_gamma(self, gamma: float) -> None:
        pass


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


class _UpCollapsible(QCollapsible):
    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        expandedIcon: QIcon | str | None = "▼",
        collapsedIcon: QIcon | str | None = "▲",
    ):
        super().__init__(title, parent, expandedIcon, collapsedIcon)
        # little hack to make the lut collapsible take up less space
        layout = cast("QVBoxLayout", self.layout())
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if (
            # look-before-leap on private attribute that may change
            hasattr(self, "_content") and (inner := self._content.layout()) is not None
        ):
            inner.setContentsMargins(0, 4, 0, 0)
            inner.setSpacing(0)

        self.setDuration(100)

        # this is a little hack to allow the buttons on the main view (below)
        # share the same row as the LUT toggle button
        layout.removeWidget(self._toggle_btn)
        self.btn_row = QHBoxLayout()
        self.btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_row.setSpacing(0)
        self.btn_row.addWidget(self._toggle_btn)
        self.btn_row.addStretch()
        layout.addLayout(self.btn_row)

    def setContent(self, content: QWidget) -> None:
        """Replace central widget (the widget that gets expanded/collapsed)."""
        self._content = content
        # this is different from upstream
        cast("QVBoxLayout", self.layout()).insertWidget(0, self._content)
        self._animation.setTargetObject(content)


# this is a PView ... but that would make a metaclass conflict
class QtViewerView(QWidget):
    currentIndexChanged = Signal()
    resetZoomClicked = Signal()
    channelModeChanged = Signal(ChannelMode)

    def __init__(
        self,
        canvas_widget: QWidget,
        histogram_widget: QWidget,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self._qcanvas = canvas_widget
        self._dims_sliders = QDimsSliders(self)
        self._dims_sliders.currentIndexChanged.connect(self.currentIndexChanged)

        # place to display dataset summary
        self._data_info_label = QElidingLabel("", parent=self)
        # place to display arbitrary text
        self._hover_info_label = QElidingLabel("", self)

        # the button that controls the display mode of the channels
        self._channel_mode_combo = QEnumComboBox(self, ChannelMode)
        self._channel_mode_combo.currentEnumChanged.connect(self.channelModeChanged)

        # button to reset the zoom of the canvas
        # TODO: unify icons across all the view frontends in a new file
        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self._set_range_btn = QPushButton(set_range_icon, "", self)
        self._set_range_btn.clicked.connect(self.resetZoomClicked)

        self._luts = _UpCollapsible(
            "LUTs",
            parent=self,
            expandedIcon=QIconifyIcon("bi:chevron-up", color="#888888"),
            collapsedIcon=QIconifyIcon("bi:chevron-down", color="#888888"),
        )
        self._btn_layout = self._luts.btn_row
        self._btn_layout.setParent(None)
        self._luts.expand()

        self._btn_layout.addWidget(self._channel_mode_combo)
        # self._btns.addWidget(self._ndims_btn)
        self._btn_layout.addWidget(self._set_range_btn)
        # self._btns.addWidget(self._add_roi_btn)

        # above the canvas
        info_widget = QWidget()
        info = QHBoxLayout(info_widget)
        info.setContentsMargins(0, 0, 0, 2)
        info.setSpacing(0)
        info.addWidget(self._data_info_label)
        info_widget.setFixedHeight(16)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(2)
        left_layout.setContentsMargins(6, 6, 6, 2)
        left_layout.addWidget(info_widget)
        left_layout.addWidget(self._qcanvas, 1)
        left_layout.addWidget(self._hover_info_label)
        left_layout.addWidget(self._dims_sliders)
        left_layout.addWidget(self._luts)
        left_layout.addLayout(self._btns)

        hist = QWidget()
        hist_layout = QVBoxLayout(hist)
        hist_layout.setSpacing(2)
        hist_layout.setContentsMargins(6, 2, 6, 6)
        hist_layout.addWidget(histogram_widget)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(left)
        splitter.addWidget(hist)
        splitter.setSizes([600, 100])

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(info_widget)
        layout.addWidget(self._qcanvas, 1)
        layout.addWidget(self._hover_info_label)
        layout.addWidget(self._dims_sliders)
        layout.addWidget(self._luts)
        layout.addLayout(self._btn_layout)

    def add_lut_view(self) -> QLUTWidget:
        wdg = QLUTWidget(self)
        self._luts.addWidget(wdg)
        return wdg

    def remove_lut_view(self, wdg: QLUTWidget) -> None:
        self._luts.removeWidget(wdg)

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

    def set_channel_mode(self, mode: ChannelMode) -> None:
        """Set the channel mode button text."""
        self._channel_mode_combo.setCurrentEnum(mode)
