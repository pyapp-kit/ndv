# /// script
# dependencies = [
#     "ndv[pyqt,pygfx]",
#     "imageio[tifffile]",
# ]
# ///
"""Example of streaming data, retaining the last N frames."""

import ndv
from ndv.models import RingBuffer

# some data we're going to stream (as if it was coming from a camera)
data = ndv.data.cells3d()[:, 1]

# a ring buffer to hold the data as it comes in
# the ring buffer is a circular buffer that holds the last N frames
N = 50
rb = RingBuffer(max_capacity=N, dtype=(data.dtype, data.shape[-2:]))

# pass the ring buffer to the viewer
viewer = ndv.ArrayViewer(rb)
viewer.show()


# function that will be called after the app is running
def stream() -> None:
    # iterate over the data, add it to the ring buffer
    for n, plane in enumerate(data):
        rb.append(plane)
        # and update the viewer index to redraw (and possibly move the slider)
        viewer.display_model.current_index.update({0: max(n, N - 1)})

        ndv.process_events()  # force viewer updates for this example


ndv.call_later(200, stream)
ndv.run_app()
