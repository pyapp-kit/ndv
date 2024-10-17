import numpy as np
from qtpy.QtWidgets import QApplication

from ndv.v2ctl import ViewerController
from ndv.v2view import ViewerView

app = QApplication([])

viewer = ViewerController(ViewerView())  # ultimately, this will be the public api
model = viewer.model
viewer.data = np.random.rand(96, 64, 128).astype(np.float32)
viewer.view.show()  # temp
app.exec()
