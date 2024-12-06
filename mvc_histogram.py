from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
)

from ndv import data
from ndv.controller._controller import HistogramController
from ndv.models import LUTModel
from ndv.models._stats import Stats

app = QApplication.instance() or QApplication([])

widget = HistogramController()
widget.stats = Stats(data.cells3d())
widget.show()

# Uncomment to attach a LUTModel
lut_model = LUTModel(
    cmap="green",
    clims=[10000, 30000],
    gamma=2,
)
widget.lut = lut_model

app.exec()
