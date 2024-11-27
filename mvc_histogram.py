from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
)

from ndv import data
from ndv.controller._controller import HistogramController

app = QApplication.instance() or QApplication([])

widget = HistogramController()
widget.data = data.cells3d()
widget.view().show()
# TODO: This throws an error when calling `DataWrapper.current_data_slice`
# (Also happens with normal Controller)
# widget.data = numpy.random.normal(10, 100, (10000, 10000))
app.exec()
