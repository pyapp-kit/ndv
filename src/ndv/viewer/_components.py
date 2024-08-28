from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import QLabel, QPushButton, QWidget
from superqt import QEnumComboBox, QIconifyIcon

if TYPE_CHECKING:
    from qtpy.QtGui import QStandardItem, QStandardItemModel

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
    RGBA = "rgba"
    MONO = "mono"

    @classmethod
    def _missing_(cls, value: object) -> ChannelMode | None:
        if value == "rgb":
            return ChannelMode.RGBA
        return None

    def __str__(self) -> str:
        return self.value


class ChannelModeCombo(QEnumComboBox):
    """A ComboBox for ChannelMode, where the RGBA enum can be removed."""

    def __init__(self, parent: QWidget | None = None, allow_rgba: bool = False):
        super().__init__(parent, enum_class=ChannelMode)
        # Find the RGBA item
        idx = list(ChannelMode.__members__.keys()).index("RGBA")
        model: QStandardItemModel = self.model()
        self._rgba_item: QStandardItem = model.item(idx)

        self.allow_rgba(allow_rgba)

    def allow_rgba(self, enable: bool) -> None:
        flags = self._rgba_item.flags()
        self._rgba_item.setFlags(
            flags | Qt.ItemFlag.ItemIsEnabled
            if enable
            else flags & ~Qt.ItemFlag.ItemIsEnabled
        )
        if self.currentEnum() == ChannelMode.RGBA and not enable:
            # Arbitrary fallback mode
            self.setCurrentEnum(ChannelMode.COMPOSITE)


class ROIButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setToolTip("Add ROI")
        self.setIcon(QIconifyIcon("mdi:vector-rectangle"))
