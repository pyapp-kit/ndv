"""Example usage of new mvc pattern."""

import numpy as np

from ndv import data, run_app
from ndv.controller import ViewerController

viewer = ViewerController()

img_data = data.cells3d()
# img_data = np.tile(img_data, (1, 1, 3, 4))
viewer.data = img_data
viewer.show()
viewer.model.current_index.update({0: 32, 1: 1})
run_app()
