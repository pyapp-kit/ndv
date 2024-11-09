"""Sample data for testing and examples."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["nd_sine_wave", "cells3d"]


def nd_sine_wave(
    shape: tuple[int, int, int, int, int] = (10, 3, 5, 512, 512),
    amplitude: float = 240,
    base_frequency: float = 5,
) -> np.ndarray:
    """5D dataset."""
    # Unpack the dimensions
    if not len(shape) == 5:
        raise ValueError("Shape must have 5 dimensions")
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


def cells3d() -> np.ndarray:
    """Load cells3d data from scikit-image."""
    try:
        from imageio.v2 import volread
    except ImportError as e:
        raise ImportError(
            "Please `pip install imageio[tifffile]` to load cells3d"
        ) from e

    url = "https://gitlab.com/scikit-image/data/-/raw/2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba/cells3d.tif"
    return volread(url)  # type: ignore [no-any-return]


def cosem_dataset(
    uri: str = "",
    dataset: str = "jrc_hela-3",
    label: str = "er-mem_pred",
    level: int = 4,
) -> Any:
    try:
        import tensorstore as ts
    except ImportError:
        raise ImportError("Please install tensorstore to fetch cosem data") from None

    if not uri:
        uri = f"{dataset}/{dataset}.n5/labels/{label}/s{level}/"

    ts_array = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "s3",
                "bucket": "janelia-cosem-datasets",
                "path": uri,
            },
        },
    ).result()
    ts_array = ts_array[ts.d[:].label["z", "y", "x"]]
    return ts_array[ts.d[("y", "x", "z")].transpose[:]]
