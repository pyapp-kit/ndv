from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QPushButton, QWidget
from superqt import QLabeledRangeSlider
from superqt.cmap import QColormapComboBox
from superqt.utils import signals_blocked

from ._dims_slider import SS

if TYPE_CHECKING:
    from collections.abc import Iterable

    import cmap

    from ndv.views.bases._graphics._canvas_elements import ImageHandle


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
        handles: Iterable[ImageHandle] = (),
        parent: QWidget | None = None,
        cmaplist: Iterable[Any] = (),
        auto_clim: bool = True,
    ) -> None:
        super().__init__(parent)
        self._handles = list(handles)
        self._name = name

        self._visible = QCheckBox(name)
        self._visible.setChecked(True)
        self._visible.toggled.connect(self._on_visible_changed)

        self._cmap = CmapCombo()
        self._cmap.currentColormapChanged.connect(self._on_cmap_changed)
        for handle in self._handles:
            self._cmap.addColormap(handle.cmap())
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
        self._auto_clim.setChecked(auto_clim)
        self._auto_clim.toggled.connect(self.update_autoscale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visible)
        layout.addWidget(self._cmap)
        layout.addWidget(self._clims)
        layout.addWidget(self._auto_clim)

        self.update_autoscale()

    def _get_state(self) -> dict[str, Any]:
        return {
            "visible": self._visible.isChecked(),
            "cmap": self._cmap.currentColormap(),
            "clims": self._clims.value(),
            "auto_clim": self._auto_clim.isChecked(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        self._visible.setChecked(state["visible"])
        self._cmap.setCurrentColormap(state["cmap"])
        self._clims.setValue(state["clims"])
        self._auto_clim.setChecked(state["auto_clim"])

    def autoscaleChecked(self) -> bool:
        return cast("bool", self._auto_clim.isChecked())

    def _on_clims_changed(self, clims: tuple[float, float]) -> None:
        self._auto_clim.setChecked(False)
        for handle in self._handles:
            handle.set_clims(clims)

    def _on_visible_changed(self, visible: bool) -> None:
        for handle in self._handles:
            handle.set_visible(visible)
        if visible:
            self.update_autoscale()

    def _on_cmap_changed(self, cmap: cmap.Colormap) -> None:
        for handle in self._handles:
            handle.set_cmap(cmap)

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
            data = handle.data()
            clims[0] = min(clims[0], np.nanmin(data))
            clims[1] = max(clims[1], np.nanmax(data))

        mi, ma = tuple(int(x) for x in clims)
        for handle in self._handles:
            handle.set_clims((mi, ma))

        # set the slider values to the new clims
        with signals_blocked(self._clims):
            self._clims.setMinimum(min(mi, self._clims.minimum()))
            self._clims.setMaximum(max(ma, self._clims.maximum()))
            self._clims.setValue((mi, ma))

    def add_handle(self, handle: ImageHandle) -> None:
        self._handles.append(handle)
        self._cmap.addColormap(handle.cmap())
        self.update_autoscale()
