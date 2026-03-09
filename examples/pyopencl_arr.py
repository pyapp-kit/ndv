# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "pyopencl",
#     "siphash24",
# ]
# ///
from __future__ import annotations

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except ImportError:
    raise ImportError("Please install pyopencl to run this example")
import ndv

# Set up OpenCL context and queue
context = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(context)


gpu_data = cl_array.to_device(queue, ndv.data.nd_sine_wave())

ndv.imshow(gpu_data)
