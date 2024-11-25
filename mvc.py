"""Example usage of new mvc pattern."""

from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

# TODO: we don't actually need to import/start a qapp anymore
# this can all be done in get_view_frontend_class
# but then we need a way to execute the app.exec() call
app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# _data = data.cosem_dataset(level=5)
_data = data.cells3d()
# _data = np.tile(_data, (2, 1, 3, 4))
_data[:, 1] = _data[:, 1] * 1
viewer.data = _data
viewer.show()
viewer.model.current_index.update({0: 32})
viewer.model.default_lut.cmap = "cubehelix"
viewer.model.channel_mode = "composite"
app.exec()
