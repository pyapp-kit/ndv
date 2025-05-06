# /// script
# dependencies = [
#     "ndv[pyqt,pygfx]",
#     "imageio[tifffile]",
# ]
# ///
"""Example of streaming data."""

import numpy as np

import ndv

# some data we're going to stream (as if it was coming from a camera)
data = ndv.data.cells3d()[:, 1]

# a buffer to hold the current frame in the viewer
buffer = np.zeros_like(data[0])
viewer = ndv.ArrayViewer(buffer)
viewer.show()


# function that will be called after the app is running
def stream(nframes: int = len(data) * 4) -> None:
    # iterate over the data, update the buffer *in place*, and update the viewer index
    for i in range(nframes):
        buffer[:] = data[i % len(data)]
        viewer.display_model.current_index.update()
        ndv.process_events()  # force viewer updates for this example


ndv.call_later(200, stream)
ndv.run_app()
