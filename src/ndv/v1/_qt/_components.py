from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy.QtCore import QSize
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import QLabel, QPushButton, QWidget
from superqt import QIconifyIcon

SPIN_GIF = str(Path(__file__).parent / "spin.gif")


class DimToggleButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        icn = QIconifyIcon("f7:view-2d", color="#333333")
        icn.addKey("f7:view-3d", state=QIconifyIcon.State.On, color="white")
        super().__init__(icn, "", parent)
        self.setCheckable(True)
        self.setChecked(True)


class QSpinner(QLabel):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        size = QSize(16, 16)
        mov = QMovie(SPIN_GIF, parent=self)
        self.setFixedSize(size)
        mov.setScaledSize(size)
        mov.setSpeed(150)
        mov.start()
        self.setMovie(mov)
        self.hide()


class ChannelMode(str, Enum):
    COMPOSITE = "composite"
    MONO = "mono"

    def __str__(self) -> str:
        return self.value


class ChannelModeButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.toggled.connect(self.next_mode)

        # set minimum width to the width of the larger string 'composite'
        self.setMinimumWidth(92)  # magic number :/

    def next_mode(self) -> None:
        if self.isChecked():
            self.setMode(ChannelMode.MONO)
        else:
            self.setMode(ChannelMode.COMPOSITE)

    def mode(self) -> ChannelMode:
        return ChannelMode.MONO if self.isChecked() else ChannelMode.COMPOSITE

    def setMode(self, mode: ChannelMode) -> None:
        # we show the name of the next mode, not the current one
        other = ChannelMode.COMPOSITE if mode is ChannelMode.MONO else ChannelMode.MONO
        self.setText(str(other))
        self.setChecked(mode == ChannelMode.MONO)


class ROIButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setToolTip("Add ROI")
        self.setIcon(QIconifyIcon("mdi:vector-rectangle"))
