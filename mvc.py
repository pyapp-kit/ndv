"""Example usage of new mvc pattern."""

import numpy as np
from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

# TODO: we don't actually need to import/start a qapp anymore
# this can all be done in get_view_frontend_class
# but then we need a way to execute the app.exec() call
app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# _data = data.cosem_dataset(level=5)
img_data = data.cells3d()
img_data = np.tile(img_data, (1, 1, 3, 4))  # just to make more of it

viewer.data = img_data
viewer.show()
viewer.model.current_index.update({0: 32})
app.exec()
