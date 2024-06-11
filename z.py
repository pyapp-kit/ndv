import sys

from qtpy.QtWidgets import QApplication

import ndv
from ndv.viewer._v2 import NDViewer

data = ndv.data.cells3d()
app = QApplication(sys.argv)
viewer = NDViewer(data)
viewer.show()
viewer.set_current_index({1: 0})
# app.exec()
