import numpy as np
from qtpy.QtWidgets import QApplication

from ndv.viewer_v2 import Viewer

app = QApplication([])

v = Viewer()
v.data = np.random.rand(96, 64, 128).astype(np.float32)
v.model.luts[1] = "viridis"
v.model.visible_axes = (-2, -1)
# print(v.model)
v.show()
v.model.current_index.update({0: 3, 1: 32, 2: 12})
app.exec()
