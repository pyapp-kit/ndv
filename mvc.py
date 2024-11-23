"""Example usage of new mvc pattern."""

from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# _data = data.cosem_dataset(level=5)
_data = data.cells3d()
# _data = np.tile(_data, (2, 1, 3, 4))
viewer.data = _data
viewer.show()
app.exec()
