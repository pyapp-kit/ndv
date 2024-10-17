from qtpy.QtWidgets import QApplication

from ndv.data import cells3d
from ndv.v2ctl import ViewerController
from ndv.v2view_qt import QViewerView

app = QApplication([])

viewer = ViewerController(QViewerView())  # ultimately, this will be the public api
model = viewer.model
viewer.data = cells3d()
viewer.view.show()  # temp
viewer.model.default_lut.cmap = "green"
app.exec()
