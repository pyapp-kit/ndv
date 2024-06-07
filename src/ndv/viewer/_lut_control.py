from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QPushButton, QWidget
from superqt import QLabeledRangeSlider
from superqt.cmap import QColormapComboBox
from superqt.utils import signals_blocked

from ._dims_slider import SS

if TYPE_CHECKING:
    import cmap

    from ._protocols import PImageHandle


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


class LutControl(QWidget):
    def __init__(
        self,
        name: str = "",
        handles: Iterable[PImageHandle] = (),
        parent: QWidget | None = None,
        cmaplist: Iterable[Any] = (),
    ) -> None:
        super().__init__(parent)
        self._handles = handles
        self._name = name

        self._visible = QCheckBox(name)
        self._visible.setChecked(True)
        self._visible.toggled.connect(self._on_visible_changed)

        self._cmap = CmapCombo()
        self._cmap.currentColormapChanged.connect(self._on_cmap_changed)
        for handle in handles:
            self._cmap.addColormap(handle.cmap)
        for color in cmaplist:
            self._cmap.addColormap(color)

        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**8)
        self._clims.valueChanged.connect(self._on_clims_changed)

        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setMaximumWidth(42)
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self.update_autoscale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visible)
        layout.addWidget(self._cmap)
        layout.addWidget(self._clims)
        layout.addWidget(self._auto_clim)

        self.update_autoscale()

    def autoscaleChecked(self) -> bool:
        return cast("bool", self._auto_clim.isChecked())

    def _on_clims_changed(self, clims: tuple[float, float]) -> None:
        self._auto_clim.setChecked(False)
        for handle in self._handles:
            handle.clim = clims

    def _on_visible_changed(self, visible: bool) -> None:
        for handle in self._handles:
            handle.visible = visible
        if visible:
            self.update_autoscale()

    def _on_cmap_changed(self, cmap: cmap.Colormap) -> None:
        for handle in self._handles:
            handle.cmap = cmap

    def update_autoscale(self) -> None:
        if (
            not self._auto_clim.isChecked()
            or not self._visible.isChecked()
            or not self._handles
        ):
            return

        # find the min and max values for the current channel
        clims = [np.inf, -np.inf]
        for handle in self._handles:
            clims[0] = min(clims[0], np.nanmin(handle.data))
            clims[1] = max(clims[1], np.nanmax(handle.data))

        mi, ma = tuple(int(x) for x in clims)
        if mi != ma:
            for handle in self._handles:
                handle.clim = (mi, ma)

            # set the slider values to the new clims
            with signals_blocked(self._clims):
                self._clims.setMinimum(min(mi, self._clims.minimum()))
                self._clims.setMaximum(max(ma, self._clims.maximum()))
                self._clims.setValue((mi, ma))
