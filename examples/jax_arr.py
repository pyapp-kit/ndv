from __future__ import annotations

try:
    import jax.numpy as jnp
except ImportError:
    raise ImportError("Please install jax to run this example")
from numpy_arr import generate_5d_sine_wave
from qtpy import QtWidgets

from ndv import NDViewer

# Example usage
array_shape = (10, 3, 5, 512, 512)  # Specify the desired dimensions
sine_wave_5d = jnp.asarray(generate_5d_sine_wave(array_shape))

if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    v = NDViewer(sine_wave_5d, channel_axis=1)
    v.show()
    qapp.exec()
