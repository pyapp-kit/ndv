# /// script
# dependencies = [
#     "ndv[pyqt,pygfx]",
#     "imageio[tifffile]",
# ]
# ///
import time

import ndv
from ndv.models import RingBuffer

cells = ndv.data.cells3d()[1]
frame_shape = cells.shape[-2:]
dtype = cells.dtype

rb = RingBuffer(max_capacity=10, dtype=(dtype, frame_shape))
viewer = ndv.ArrayViewer(rb)
viewer.show()


def stream() -> None:
    for plane in cells:
        rb.append(plane)
        time.sleep(0.01)
        # bit of a hack to force updates for this example
        viewer._app.process_events()


ndv.call_later(200, stream)
ndv.run_app()
