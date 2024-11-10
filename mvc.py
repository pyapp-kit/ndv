"""Example usage of new mvc pattern."""

import numpy as np
from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# viewer.data = data.cosem_dataset(level=5)
viewer.data = np.tile(data.cells3d(), (2, 1, 3, 4))
viewer._view.show()
app.exec()
