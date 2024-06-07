from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QFileDialog, QPushButton, QWidget
from superqt.iconify import QIconifyIcon

if TYPE_CHECKING:
    from ._indexing import DataWrapper


class SaveButton(QPushButton):
    def __init__(
        self,
        data_wrapper: DataWrapper,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.setIcon(QIconifyIcon("mdi:content-save"))
        self.clicked.connect(self._on_click)

        self._data_wrapper = data_wrapper
        self._last_loc = str(Path.home())

    def _on_click(self) -> None:
        self._last_loc, _ = QFileDialog.getSaveFileName(
            self, "Choose destination", str(self._last_loc), ""
        )
        suffix = Path(self._last_loc).suffix
        if suffix in (".zarr", ".ome.zarr", ""):
            self._data_wrapper.save_as_zarr(self._last_loc)
        else:
            raise ValueError(f"Unsupported file format: {self._last_loc}")
