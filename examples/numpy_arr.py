from __future__ import annotations

import numpy as np


def generate_5d_sine_wave(
    shape: tuple[int, int, int, int, int],
    amplitude: float = 240,
    base_frequency: float = 5,
) -> np.ndarray:
    """5D dataset."""
    # Unpack the dimensions
    angle_dim, freq_dim, phase_dim, ny, nx = shape

    # Create an empty array to hold the data
    output = np.zeros(shape)

    # Define spatial coordinates for the last two dimensions
    half_per = base_frequency * np.pi
    x = np.linspace(-half_per, half_per, nx)
    y = np.linspace(-half_per, half_per, ny)
    y, x = np.meshgrid(y, x)

    # Iterate through each parameter in the higher dimensions
    for phase_idx in range(phase_dim):
        for freq_idx in range(freq_dim):
            for angle_idx in range(angle_dim):
                # Calculate phase and frequency
                phase = np.pi / phase_dim * phase_idx
                frequency = 1 + (freq_idx * 0.1)  # Increasing frequency with each step

                # Calculate angle
                angle = np.pi / angle_dim * angle_idx
                # Rotate x and y coordinates
                xr = np.cos(angle) * x - np.sin(angle) * y
                np.sin(angle) * x + np.cos(angle) * y

                # Compute the sine wave
                sine_wave = (amplitude * 0.5) * np.sin(frequency * xr + phase)
                sine_wave += amplitude * 0.5

                # Assign to the output array
                output[angle_idx, freq_idx, phase_idx] = sine_wave

    return output


try:
    from skimage import data

    img = data.cells3d()
except Exception:
    img = generate_5d_sine_wave((10, 3, 8, 512, 512))


if __name__ == "__main__":
    from qtpy import QtWidgets

    from ndv import NDViewer

    qapp = QtWidgets.QApplication([])
    v = NDViewer(img)
    v.show()
    qapp.exec()
