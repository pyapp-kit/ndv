from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ndv.histogram.model import StatsModel
from ndv.histogram.views._vispy import VispyHistogramView

if TYPE_CHECKING:
    from typing import Any


class Controller:
    """A (Qt) wrapper around another HistogramView with some additional controls."""

    def __init__(self) -> None:
        self._wdg = QWidget()
        self._model = StatsModel()
        self._view = VispyHistogramView()

        # A HistogramView is both a StatsView and a LUTView
        # StatModel <-> StatsView
        self._model.events.histogram.connect(
            lambda data: self._view.set_histogram(*data)
        )
        # LutModel <-> LutView (TODO)
        # LutView -> LutModel (TODO: Currently LutView <-> LutView)
        self._view.gammaChanged.connect(self._view.set_gamma)
        self._view.climsChanged.connect(self._view.set_clims)

        # Vertical box
        self._vert = QPushButton("Vertical")
        self._vert.setCheckable(True)
        self._vert.toggled.connect(self._view.set_vertical)

        # Log box
        self._log = QPushButton("Logarithmic")
        self._log.setCheckable(True)
        self._log.toggled.connect(self._view.set_range_log)

        # Data updates
        self._data_btn = QPushButton("Change Data")
        self._data_btn.setCheckable(True)
        self._data_btn.toggled.connect(
            lambda toggle: self.timer.blockSignals(not toggle)
        )

        def _update_data() -> None:
            """Replaces the displayed data."""
            self._model.data = np.random.normal(10, 10, 10000)

        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.blockSignals(True)
        self.timer.timeout.connect(_update_data)
        self.timer.start()

        # Layout
        self._layout = QVBoxLayout(self._wdg)
        self._layout.addWidget(self._view.view())
        self._layout.addWidget(self._vert)
        self._layout.addWidget(self._log)
        self._layout.addWidget(self._data_btn)

    def view(self) -> Any:
        """Returns an object that can be displayed by the active backend."""
        return self._wdg


app = QApplication.instance() or QApplication([])

widget = Controller()
widget.view().show()
app.exec()
