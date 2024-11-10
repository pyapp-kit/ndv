"""Example usage of new mvc pattern."""

from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# with suppress(ImportError):
viewer.data = data.cosem_dataset(level=5)
# viewer.data = data.cells3d()
viewer._view.show()
app.exec()
