from __future__ import annotations

import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import psygnal
from qtpy.QtCore import QObject, QPoint, QSize, Qt, Signal
from qtpy.QtGui import QCursor, QMouseEvent, QMovie
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import (
    QCollapsible,
    QLabeledRangeSlider,
    QLabeledSlider,
)
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv._types import AxisKey
from ndv.models._array_display_model import ChannelMode
from ndv.models._lut_model import ClimPolicy, ClimsManual, ClimsPercentile
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views.bases import ArrayView, LutView

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap
    from psygnal import EmissionInfo
    from qtpy.QtGui import QIcon

    from ndv._types import AxisKey, ChannelKey
    from ndv.views.bases._graphics._canvas import HistogramCanvas
    from ndv.views.bases._graphics._canvas_elements import (
        CanvasElement,
        RectangularROIHandle,
    )

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


class _CmapCombo(QColormapComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, allow_user_colormaps=True, add_colormap_text="Add...")
        self.setMinimumSize(140, 21)
        # self.setStyleSheet("background-color: transparent;")

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.findChild(QFrame)
        popup.setMinimumWidth(self.width() + 100)
        popup.move(popup.x(), popup.y() - self.height() - popup.height())

    # TODO: upstream me
    def setCurrentColormap(self, cmap_: cmap.Colormap) -> None:
        """Adds the color to the QComboBox and selects it."""
        for idx in range(self.count()):
            if item := self.itemColormap(idx):
                if item.name == cmap_.name:
                    # cmap_ is already here - just select it
                    self.setCurrentIndex(idx)
                    return

        # cmap_ not in the combo box - add it!
        self.addColormap(cmap_)
        # then, select it!
        # NB: "Add..." was at idx, now it's at idx+1 and cmap_ is at idx
        self.setCurrentIndex(idx)


class _QSpinner(QLabel):
    SPIN_GIF = str(Path(__file__).parent.parent / "_resources" / "spin.gif")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        size = QSize(18, 18)
        mov = QMovie(self.SPIN_GIF, parent=self)
        self.setFixedSize(size)
        mov.setSpeed(150)
        mov.start()
        self.setMovie(mov)

        # make semi-transparent
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.6)
        self.setGraphicsEffect(effect)


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
        self.setStyleSheet(r"QPushButton {border: none; padding: 0; margin: 0;}")
        self._popup = QtPopup(self)
        form = QFormLayout(self._popup.frame)
        form.setContentsMargins(6, 6, 6, 6)
        form.addRow("FPS", self.spin)

    def mousePressEvent(self, e: QMouseEvent | None) -> None:
        if e and e.button() == Qt.MouseButton.RightButton:
            self._show_fps_dialog()
        else:
            super().mousePressEvent(e)

    def _show_fps_dialog(self) -> None:
        self._popup.show_above_mouse()


class _DimToggleButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        icn = QIconifyIcon("f7:view-2d", color="#333333")
        icn.addKey("f7:view-3d")
        super().__init__(icn, "", parent)
        self.setCheckable(True)


class _QLUTWidget(QWidget):
    def __init__(
        self,
        default_luts: Sequence[Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # -- WIDGETS -- #
        self.visible = QCheckBox()

        self.cmap = _CmapCombo()
        self.cmap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cmap.addColormaps(default_luts)

        self.clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)

        WHITE_SS = SLIDER_STYLE + "SliderLabel { font-size: 10px; color: white;}"
        self.clims.setStyleSheet(WHITE_SS)
        self.clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self.clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self.clims.setRange(0, 2**16)

        self.auto_clim = QPushButton("Auto")
        self.auto_clim.setMaximumWidth(42)
        self.auto_clim.setCheckable(True)

        add_histogram_icon = QIconifyIcon("foundation:graph-bar")
        self.histogram_btn = QPushButton(add_histogram_icon, "")
        self.histogram_btn.setCheckable(True)

        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self.hist_range = QPushButton(set_range_icon, "", self)
        self.hist_range.setVisible(False)

        log_icon = QIconifyIcon("mdi:math-log")
        self.hist_log = QPushButton(log_icon, "", self)
        self.hist_log.setToolTip("log (base 10, count+1)")
        self.hist_log.setCheckable(True)
        self.hist_log.setVisible(False)

        self.upper_tail = QDoubleSpinBox()
        self.upper_tail.setRange(0, 49.9)
        self.upper_tail.setSingleStep(0.1)
        self.upper_tail.setSuffix("%")
        self.upper_tail.setValue(0)

        self.lower_tail = QDoubleSpinBox()
        self.lower_tail.setRange(0, 49.9)
        self.lower_tail.setSingleStep(0.1)
        self.lower_tail.setSuffix("%")
        self.lower_tail.setValue(0)

        self.auto_popup = QtPopup(self)
        form = QFormLayout(self.auto_popup.frame)
        form.setContentsMargins(6, 6, 6, 6)
        form.addRow("Exclude Darkest:", self.lower_tail)
        form.addRow("Exclude Brightest:", self.upper_tail)

        # -- LAYOUT -- #

        # "main" lut controls (always visible)
        self._lut_layout = QHBoxLayout()
        self._lut_layout.setSpacing(5)
        self._lut_layout.setContentsMargins(0, 0, 0, 0)
        self._lut_layout.addWidget(self.visible)
        self._lut_layout.addWidget(self.cmap)
        self._lut_layout.addWidget(self.clims)
        self._lut_layout.addWidget(self.auto_clim)
        self._lut_layout.addWidget(self.histogram_btn)

        # histogram controls go in their own layout
        hist_ctrls_layout = QVBoxLayout()
        # Add a vertical spacer that expands to take up available space
        # This is the key component that pushes everything down
        hist_ctrls_layout.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        hist_ctrls_layout.addWidget(self.hist_log)
        hist_ctrls_layout.addWidget(self.hist_range)

        # histogram layout contains controls + a histogram (which is added later)
        self.hist_layout = QHBoxLayout()
        self.hist_layout.addLayout(hist_ctrls_layout)

        # Overall layout
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(self._lut_layout)
        self._layout.addLayout(self.hist_layout)


class QLutView(LutView):
    # NB: In practice this will be a ChannelKey but Unions not allowed here.
    histogramRequested = psygnal.Signal(object)

    def __init__(
        self,
        channel: ChannelKey = None,
        default_luts: Sequence[Any] = ("gray", "green", "magenta"),
    ) -> None:
        super().__init__()
        self._qwidget = _QLUTWidget(default_luts)
        self._channel = channel
        self.histogram: HistogramCanvas | None = None
        # TODO: use emit_fast
        self._qwidget.histogram_btn.toggled.connect(self._on_q_histogram_toggled)
        self._qwidget.hist_log.toggled.connect(self._on_log_btn_toggled)
        self._qwidget.hist_range.clicked.connect(self._on_set_histogram_range_clicked)
        self._qwidget.visible.toggled.connect(self._on_q_visibility_changed)
        self._qwidget.cmap.currentColormapChanged.connect(self._on_q_cmap_changed)
        self._qwidget.clims.valueChanged.connect(self._on_q_clims_changed)
        self._qwidget.auto_clim.toggled.connect(self._on_q_auto_changed)
        self._qwidget.lower_tail.valueChanged.connect(self._on_q_tails_changed)
        self._qwidget.upper_tail.valueChanged.connect(self._on_q_tails_changed)

        self._qwidget.auto_clim.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._qwidget.auto_clim.customContextMenuRequested.connect(
            self._show_autoscale_popup
        )

    def frontend_widget(self) -> QWidget:
        return self._qwidget

    def set_channel_name(self, name: str) -> None:
        self._qwidget.visible.setText(name)

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        self._qwidget.auto_clim.setChecked(not policy.is_manual)
        if isinstance(policy, ClimsPercentile):
            self._qwidget.lower_tail.setValue(policy.min_percentile)
            self._qwidget.upper_tail.setValue(100 - policy.max_percentile)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        self._qwidget.cmap.setCurrentColormap(cmap)

    def set_clims(self, clims: tuple[float, float]) -> None:
        # block self._qwidget._clims, otherwise autoscale will be forced off
        with signals_blocked(self._qwidget.clims):
            self._qwidget.clims.setValue(clims)

    def set_clim_bounds(
        self,
        bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        # block self._qwidget._clims, otherwise autoscale will be forced off
        mi = 0 if bounds[0] is None else int(bounds[0])
        ma = 65535 if bounds[1] is None else int(bounds[1])
        with signals_blocked(self._qwidget.clims):
            self._qwidget.clims.setRange(mi, ma)

    def set_gamma(self, gamma: float) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        self._qwidget.visible.setChecked(visible)

    def set_visible(self, visible: bool) -> None:
        self._qwidget.setVisible(visible)

    def close(self) -> None:
        self._qwidget.close()

    def _on_q_visibility_changed(self, visible: bool) -> None:
        if self._model:
            self._model.visible = visible

    def _on_q_cmap_changed(self, cmap: cmap.Colormap) -> None:
        if self._model:
            self._model.cmap = cmap

    def _on_q_clims_changed(self, clims: tuple[float, float]) -> None:
        if self._model:
            self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_q_auto_changed(self, autoscale: bool) -> None:
        if self._model:
            if autoscale:
                lower_tail = self._qwidget.lower_tail.value()
                upper_tail = self._qwidget.upper_tail.value()
                self._model.clims = ClimsPercentile(
                    min_percentile=lower_tail, max_percentile=100 - upper_tail
                )
            else:
                clims = self._qwidget.clims.value()
                self._model.clims = ClimsManual(min=clims[0], max=clims[1])

    def _on_q_tails_changed(self, value: float) -> None:
        self._on_q_auto_changed(self._qwidget.auto_clim.isChecked())

    def _on_q_histogram_toggled(self, toggled: bool) -> None:
        # Recursively show/hide hist_layout
        self._set_layout_visibility(toggled, self._qwidget.hist_layout)
        # Request histogram if not created yet
        if self.histogram is None:
            self.histogramRequested.emit(self._channel)

    def _set_layout_visibility(self, toggled: bool, layout: QLayout) -> None:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            if wdg := item.widget():
                wdg.setVisible(toggled)
            elif child_layout := item.layout():
                self._set_layout_visibility(toggled, child_layout)

    def _on_log_btn_toggled(self, toggled: bool) -> None:
        if hist := self.histogram:
            hist.set_log_base(10 if toggled else None)

    def _on_set_histogram_range_clicked(self) -> None:
        self._qwidget.hist_log.setChecked(False)
        if hist := self.histogram:
            hist.set_range()

    def _add_histogram(self, histogram: HistogramCanvas) -> None:
        # Add widget to view
        self.histogram = histogram
        widget = cast("QWidget", histogram.frontend_widget())
        # Resize widget to a respectable size
        widget.setFixedHeight(100)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._qwidget.hist_layout.addWidget(widget, 1)

    def _show_autoscale_popup(self) -> None:
        self._qwidget.auto_popup.show_above_mouse()


class QRGBView(QLutView):
    def __init__(self, channel: ChannelKey = None) -> None:
        super().__init__(channel)
        # Hide the cmap selector
        self._qwidget.cmap.setVisible(False)
        # Insert a new label
        self._label = QLabel("RGB")
        self._qwidget._lut_layout.insertWidget(1, self._label)


class ROIButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setToolTip("Add ROI")
        self.setIcon(QIconifyIcon("mdi:vector-rectangle"))


class DimRow(QObject):
    def __init__(
        self, axis: AxisKey, _coords: Sequence, parent: QObject | None
    ) -> None:
        super().__init__(parent)
        self.slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.index_label = self.slider._label
        self.play_btn = PlayButton()
        self.play_btn.fpsChanged.connect(self.set_fps)
        self.play_btn.toggled.connect(self.set_animated)
        self.label = QLabel(str(axis))
        self.out_of = QLabel(f"/ {len(_coords) - 1}")
        self.out_of.setStyleSheet("margin: 0 0 2px 0;")  # hack

        self._timer_id: int | None = None

    def set_fps(self, fps: float) -> None:
        self.play_btn.spin.setValue(fps)
        self.set_animated(self.play_btn.isChecked())

    def set_animated(self, animate: bool) -> None:
        if animate:
            if self._timer_id is not None:
                self.killTimer(self._timer_id)
            interval = int(1000 / self.play_btn.spin.value())
            self._timer_id = self.startTimer(interval)
            self.play_btn.setChecked(True)
        elif self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None
            self.play_btn.setChecked(False)

    def timerEvent(self, event: Any) -> None:
        """Handle timer event for play button, move to the next frame."""
        # TODO
        # for now just increment the value by 1, but we should be able to
        # take FPS into account better and skip additional frames if the timerEvent
        # is delayed for some reason.
        inc = 1
        ival = self.slider.value()
        ival = (ival + inc) % (self.slider.maximum() + 1)
        self.slider.setValue(ival)


class _QDimsSliders(QWidget):
    currentIndexChanged = Signal()

    _rPLAY_BTN = 0
    _rLABEL = 1
    _rSLIDER = 2
    _rINDEX = 3
    _rTOT = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}
        self._play_btn_visible: bool = True
        self.setStyleSheet(SLIDER_STYLE)

        self._layout = QGridLayout(self)
        self._layout.setSpacing(1)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        grid = self._layout
        for axis, _coords in coords.items():
            # Create a slider for axis if necessary
            if axis not in self._sliders:
                row = grid.rowCount()
                dim_row = DimRow(axis, _coords, self)
                dim_row.slider.valueChanged.connect(self.currentIndexChanged)
                if not self._play_btn_visible:
                    dim_row.play_btn.setVisible(False)
                grid.addWidget(dim_row.play_btn, row, self._rPLAY_BTN)
                grid.addWidget(dim_row.label, row, self._rLABEL)
                grid.addWidget(dim_row.slider, row, self._rSLIDER)
                grid.addWidget(
                    dim_row.index_label, row, self._rINDEX, Qt.AlignmentFlag.AlignRight
                )
                grid.addWidget(
                    dim_row.out_of, row, self._rTOT, Qt.AlignmentFlag.AlignLeft
                )
                self._sliders[axis] = dim_row.slider

            # Update axis slider with coordinates
            sld = self._sliders[axis]
            if isinstance(_coords, range):
                step = _coords.step
                start, stop = _coords.start, (_coords.stop - 1)
            else:
                step = 1
                start, stop = 0, len(_coords) - 1
            sld.setSingleStep(step)
            sld.setRange(start, stop)
            self.setRowTotal(sld, stop)

        self.currentIndexChanged.emit()

    def hide_dimensions(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                self.setRowVisible(slider, False)
            elif show_remainder:
                self.setRowVisible(slider, True)

    def _getSliderRow(self, slider: QWidget) -> int | None:
        # Find the row of the given slider
        for r in range(self._layout.rowCount()):
            if item := self._layout.itemAtPosition(r, self._rSLIDER):
                if item.widget() is slider:
                    return r
        return None

    def setRowTotal(self, slider: QWidget, total: int) -> None:
        if (sr := self._getSliderRow(slider)) is None:
            return

        # Update the total label for the given row
        item = self._layout.itemAtPosition(sr, self._rTOT)
        if item and (label := item.widget()):
            cast("QLabel", label).setText(f"/ {total}")

    def setRowVisible(self, slider: QWidget, visible: bool) -> None:
        if (sr := self._getSliderRow(slider)) is None:
            return

        # Toggle visibility of all widgets in the found row
        for c in range(self._layout.columnCount()):
            if c == self._rPLAY_BTN and not self._play_btn_visible:
                # don't show play button when not visible
                continue
            item = self._layout.itemAtPosition(sr, c)
            if item and (widget := item.widget()):
                widget.setVisible(visible)
                if isinstance(widget, PlayButton) and not visible:
                    widget.setChecked(False)

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

    def set_play_button_visible(self, visible: bool) -> None:
        """Set the visibility of the play button on all rows."""
        self._play_btn_visible = visible
        for row in range(self._layout.rowCount()):
            item = self._layout.itemAtPosition(row, self._rSLIDER)
            if item and (wdg := item.widget()) and not wdg.isVisible():
                continue  # pragma: no cover, skip if the row is hidden
            item = self._layout.itemAtPosition(row, self._rPLAY_BTN)
            if item and (btn := item.widget()):
                btn.setVisible(visible)

    def stop_animations(self) -> None:
        """Stop all animations on the play buttons."""
        for child in self.findChildren(DimRow):
            child.set_animated(False)


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
class _QArrayViewer(QWidget):
    def __init__(self, canvas_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        self._canvas_widget = canvas_widget
        self.dims_sliders = _QDimsSliders(self)

        # place to display dataset summary
        self.data_info_label = QLabel("", parent=self)
        # place to display arbitrary text
        self.hover_info_label = QLabel("", self)

        # spinner to indicate progress
        self._progress_spinner = _QSpinner(canvas_widget)
        self._progress_spinner.hide()

        # the button that controls the display mode of the channels
        # not using QEnumComboBox because we want to exclude some values for now
        self.channel_mode_combo = QComboBox(self)
        self.channel_mode_combo.addItems(
            [
                ChannelMode.GRAYSCALE.value,
                ChannelMode.COMPOSITE.value,
                ChannelMode.RGBA.value,
            ]
        )

        # button to reset the zoom of the canvas
        # TODO: unify icons across all the view frontends in a new file
        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self.set_range_btn = QPushButton(set_range_icon, "", self)

        # button to draw ROIs
        self._roi_handle: RectangularROIHandle | None = None
        self._selection: CanvasElement | None = None
        self.add_roi_btn = ROIButton()

        self.luts = _UpCollapsible(
            "LUTs",
            parent=self,
            expandedIcon=QIconifyIcon("bi:chevron-up", color="#888888"),
            collapsedIcon=QIconifyIcon("bi:chevron-down", color="#888888"),
        )
        self._btn_layout = self.luts.btn_row
        self._btn_layout.setParent(None)
        self.luts.expand()

        # button to change number of displayed dimensions
        self.ndims_btn = _DimToggleButton(self)

        self._btn_layout.addWidget(self.channel_mode_combo)
        self._btn_layout.addWidget(self.ndims_btn)
        self._btn_layout.addWidget(self.add_roi_btn)
        self._btn_layout.addWidget(self.set_range_btn)

        self._btns = QWidget()
        self._btns.setLayout(self._btn_layout)

        # above the canvas
        self._info_widget = QWidget()
        self._info_widget.setFixedHeight(16)
        info = QHBoxLayout(self._info_widget)
        self.data_info_label.setMaximumWidth(400)
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(0)
        info.addWidget(self.data_info_label)
        info.addStretch()
        info.addWidget(self.hover_info_label)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(2)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._info_widget)
        left_layout.addWidget(canvas_widget, 1)
        left_layout.addWidget(self.dims_sliders)
        left_layout.addWidget(self.luts)
        left_layout.addWidget(self._btns)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.addWidget(left)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.splitter)

    def resizeEvent(self, a0: Any) -> None:
        # position at spinner the top right of the canvas_widget:
        canv, spinner = self._canvas_widget, self._progress_spinner
        pad = 4
        spinner.move(canv.width() - spinner.width() - pad, pad)
        super().resizeEvent(a0)

    def closeEvent(self, a0: Any) -> None:
        with suppress(AttributeError):
            del self._canvas_widget
        super().closeEvent(a0)


class QtArrayView(ArrayView):
    def __init__(
        self,
        canvas_widget: QWidget,
        viewer_model: ArrayViewerModel,
    ) -> None:
        self._viewer_model = viewer_model
        self._qwidget = qwdg = _QArrayViewer(canvas_widget)
        # Mapping of channel key to LutViews
        self._luts: dict[ChannelKey, QLutView] = {}
        qwdg.add_roi_btn.toggled.connect(self._on_add_roi_clicked)

        self._viewer_model.events.connect(self._on_viewer_model_event)

        # TODO: use emit_fast
        qwdg.dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)
        qwdg.channel_mode_combo.currentTextChanged.connect(
            self._on_channel_mode_changed
        )
        qwdg.set_range_btn.clicked.connect(self.resetZoomClicked.emit)
        qwdg.ndims_btn.toggled.connect(self._on_ndims_toggled)

        self._visible_axes: Sequence[AxisKey] = []

    def add_lut_view(self, channel: ChannelKey) -> QLutView:
        view = (
            QRGBView(channel)
            if channel == "RGB"
            else QLutView(channel, self._viewer_model.default_luts)
        )
        self._luts[channel] = view

        view.histogramRequested.connect(self.histogramRequested)
        self._qwidget.luts.addWidget(view.frontend_widget())
        return view

    def remove_lut_view(self, view: LutView) -> None:
        self._qwidget.luts.removeWidget(cast("QLutView", view).frontend_widget())

    def _on_channel_mode_changed(self, text: str) -> None:
        self.channelModeChanged.emit(ChannelMode(text))

    def add_histogram(self, channel: ChannelKey, histogram: HistogramCanvas) -> None:
        if lut := self._luts.get(channel, None):
            lut._add_histogram(histogram)

    def remove_histogram(self, widget: QWidget) -> None:
        widget.setParent(None)
        widget.deleteLater()

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        self._qwidget.dims_sliders.create_sliders(coords)

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        """Hide sliders based on visible axes."""
        self._qwidget.dims_sliders.hide_dimensions(axes_to_hide, show_remainder)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return self._qwidget.dims_sliders.current_index()

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        self._qwidget.dims_sliders.set_current_index(value)

    def _on_ndims_toggled(self, is_3d: bool) -> None:
        self._qwidget.dims_sliders.stop_animations()
        self.nDimsRequested.emit(3 if is_3d else 2)

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes  # no widget to control this yet

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._qwidget.ndims_btn.setChecked(len(axes) > 2)

    def set_data_info(self, text: str) -> None:
        """Set the data info text, above the canvas."""
        self._qwidget.data_info_label.setText(text)

    def set_hover_info(self, text: str) -> None:
        """Set the hover info text, below the canvas."""
        self._qwidget.hover_info_label.setText(text)

    def set_channel_mode(self, mode: ChannelMode) -> None:
        """Set the channel mode button text."""
        self._qwidget.channel_mode_combo.setCurrentText(mode.value)

    def set_visible(self, visible: bool) -> None:
        self._qwidget.setVisible(visible)

    def close(self) -> None:
        self._qwidget.close()

    def frontend_widget(self) -> QWidget:
        return self._qwidget

    def _on_add_roi_clicked(self, checked: bool) -> None:
        self._viewer_model.interaction_mode = (
            InteractionMode.CREATE_ROI if checked else InteractionMode.PAN_ZOOM
        )

    def _on_viewer_model_event(self, info: EmissionInfo) -> None:
        sig_name = info.signal.name
        value = info.args[0]
        if sig_name == "show_progress_spinner":
            self._qwidget._progress_spinner.setVisible(value)
        if sig_name == "interaction_mode":
            # If leaving CanvasMode.CREATE_ROI, uncheck the ROI button
            new, old = info.args
            if old == InteractionMode.CREATE_ROI:
                self._qwidget.add_roi_btn.setChecked(False)
        elif sig_name == "show_histogram_button":
            for lut in self._luts.values():
                lut._qwidget.histogram_btn.setVisible(value)
        elif sig_name == "show_roi_button":
            self._qwidget.add_roi_btn.setVisible(value)
        elif sig_name == "show_channel_mode_selector":
            self._qwidget.channel_mode_combo.setVisible(value)
        elif sig_name == "show_reset_zoom_button":
            self._qwidget.set_range_btn.setVisible(value)
        elif sig_name == "show_3d_button":
            self._qwidget.ndims_btn.setVisible(value)
        elif sig_name == "show_play_button":
            self._qwidget.dims_sliders.set_play_button_visible(value)
        elif sig_name == "show_data_info":
            self._qwidget._info_widget.setVisible(value)
        elif sig_name == "show_controls":
            # Show or hide the entire controls area (dims sliders + LUTs)
            self._qwidget.luts.setVisible(value)
            self._qwidget.dims_sliders.setVisible(value)
            self._qwidget._btns.setVisible(value)
