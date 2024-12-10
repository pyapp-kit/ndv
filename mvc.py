"""Example usage of new mvc pattern."""

from ndv import data, run_app
from ndv.controller import ViewerController

viewer = ViewerController()

# _data = data.cosem_dataset(level=5)
_data = data.cells3d()
# _data = np.tile(_data, (2, 1, 3, 4))
viewer.data = _data
viewer.show()
viewer.model.current_index.update({0: 32})
viewer.model.channel_mode = "composite"
run_app()
