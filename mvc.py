from qtpy.QtWidgets import QApplication

from ndv.controller import ViewerController
from ndv.data import cells3d
from ndv.v2view_qt import QViewerView

app = QApplication([])

viewer = ViewerController(QViewerView())  # ultimately, this will be the public api
viewer.data = cells3d()
viewer.view.show()
app.exec()
