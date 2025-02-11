import numpy as np

import ndv
from ndv import StreamingViewer

viewer = StreamingViewer()
viewer.show()
shape, dtype = (100, 100), "uint8"
viewer.setup(shape, dtype)
viewer.set_data(np.random.randint(0, 255, shape).astype(dtype))
ndv.run_app()
