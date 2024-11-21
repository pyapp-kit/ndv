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

from ndv.models import LUTModel, StatsModel
from ndv.views._vispy._vispy import VispyHistogramView

if TYPE_CHECKING:
    from typing import Any

    from ndv.views.protocols import PHistogramView


class Controller:
    """A (Qt) wrapper around another HistogramView with some additional controls."""

    def __init__(
        self,
        stats_model: StatsModel | None = None,
        lut_model: LUTModel | None = None,
        view: PHistogramView | None = None,
    ) -> None:
        self._wdg = QWidget()
        if stats_model is None:
            stats_model = StatsModel()
        if lut_model is None:
            lut_model = LUTModel()
        if view is None:
            view = VispyHistogramView()
        self._stats = stats_model
        self._lut = lut_model
        self._view = view

        # A HistogramView is both a StatsView and a LUTView
        # StatModel <-> StatsView
        self._stats.events.data.connect(self._set_data)
        self._stats.events.bins.connect(self._set_data)
        # LutModel <-> LutView
        self._lut.events.clims.connect(self._set_model_clims)
        self._view.climsChanged.connect(self._set_view_clims)
        self._lut.events.gamma.connect(self._set_model_gamma)
        self._view.gammaChanged.connect(self._set_view_gamma)

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
            self._stats.data = np.random.normal(10, 10, 10000)

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

    def _set_data(self) -> None:
        values, bin_edges = self._stats.histogram
        self._view.set_histogram(values, bin_edges)

    def _set_model_clims(self) -> None:
        clims = self._lut.clims
        self._view.set_clims(clims)

    def _set_view_clims(self, clims: tuple[float, float]) -> None:
        self._lut.clims = clims

    def _set_model_gamma(self) -> None:
        gamma = self._lut.gamma
        self._view.set_gamma(gamma)

    def _set_view_gamma(self, gamma: float) -> None:
        self._lut.gamma = gamma

    def view(self) -> Any:
        """Returns an object that can be displayed by the active backend."""
        return self._wdg


app = QApplication.instance() or QApplication([])

widget = Controller()
widget.view().show()
app.exec()
