"""Example usage of new mvc pattern."""

from qtpy.QtWidgets import QApplication

from ndv.controller import ViewerController
from ndv.data import cells3d

app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api
viewer.data = cells3d()
viewer._view.show()
app.exec()
