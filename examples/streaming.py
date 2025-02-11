import time

import numpy as np

# unfortunate leaking of abstraction... to schedule a callback
from qtpy.QtCore import QTimer

import ndv
from ndv import StreamingViewer
from ndv.models._lut_model import LUTModel

viewer = StreamingViewer()
viewer.show()

shape, dtype = (1024, 1024), "uint8"
viewer.setup(shape, dtype, channels={0: LUTModel(cmap="magma")})


def stream(n: int = 50) -> None:
    for _ in range(n):
        viewer.set_data(np.random.randint(0, 255, shape).astype(dtype))
        time.sleep(0.01)


QTimer.singleShot(1000, stream)
ndv.run_app()
