from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ndv.histogram._view import HistogramView

if TYPE_CHECKING:
    from typing import Any

    from ndv.histogram._vispy import VispyHistogramView


class QtHistogramView(HistogramView):
    def __init__(self, histogram: VispyHistogramView) -> None:
        self._wdg = QWidget()
        self._hist = histogram

        # Vertical box
        self._vert = QPushButton()
        self._vert.setText("Vertical")
        self._vert.setCheckable(True)
        self._vert.toggled.connect(self.set_vertical)

        # Log box
        self._log = QPushButton()
        self._log.setText("Logarithmic")
        self._log.setCheckable(True)
        self._log.toggled.connect(self.enable_range_log)

        self._layout = QVBoxLayout(self._wdg)
        self._layout.addWidget(histogram.view())
        self._layout.addWidget(self._vert)
        self._layout.addWidget(self._log)

    # -- Protocol methods -- #

    def view(self) -> Any:
        return self._wdg

    def set_vertical(self, vertical: bool) -> None:
        self._hist.set_vertical(vertical)

    def enable_range_log(self, enabled: bool) -> None:
        self._hist.enable_range_log(enabled)
