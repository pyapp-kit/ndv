from __future__ import annotations

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except ImportError:
    raise ImportError("Please install jax to run this example")
from numpy_arr import generate_5d_sine_wave

# Set up OpenCL context and queue
context = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(context)


# Example usage
array_shape = (10, 3, 5, 512, 512)  # Specify the desired dimensions
sine_wave_5d = generate_5d_sine_wave(array_shape)
cl_sine_wave = cl_array.to_device(queue, sine_wave_5d)


if __name__ == "__main__":
    from qtpy import QtWidgets

    from ndv import NDViewer

    qapp = QtWidgets.QApplication([])
    v = NDViewer(cl_sine_wave, channel_axis=1)
    v.show()
    qapp.exec()
