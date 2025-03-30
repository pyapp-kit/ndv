"""Sample data for testing and examples."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["astronaut", "cat", "cells3d", "cosem_dataset", "nd_sine_wave"]


def nd_sine_wave(
    shape: tuple[int, int, int, int, int] = (10, 3, 5, 512, 512),
    amplitude: float = 240,
    base_frequency: float = 5,
) -> np.ndarray:
    """5D dataset: `(10, 3, 5, 512, 512)`, float64."""
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

    return output.astype(np.float32)


def cells3d() -> np.ndarray:
    """Load cells3d from scikit-image `(60, 2, 256, 256)` uint16.

    Requires `imageio and tifffile` to be installed.
    """
    try:
        from imageio.v2 import volread
    except ImportError as e:
        raise ImportError(
            "Please `pip install imageio[tifffile]` to load cells3d"
        ) from e

    url = "https://gitlab.com/scikit-image/data/-/raw/2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba/cells3d.tif"
    data = np.asarray(volread(url))

    # this data has been stretched to 16 bit, and lacks certain intensity values
    # add a small random integer to each pixel ... so the histogram is not silly
    data = (data + np.random.randint(-24, 24, data.shape)).clip(0, 65535)
    return data.astype(np.uint16)


def cat() -> np.ndarray:
    """Load RGB cat data `(300, 451, 3)`, uint8.

    Requires [imageio](https://pypi.org/project/imageio/) to be installed.
    """
    return _imread("imageio:chelsea.png")


def astronaut() -> np.ndarray:
    """Load RGB data `(512, 512, 3)`, uint8.

    Requires [imageio](https://pypi.org/project/imageio/) to be installed.
    """
    return _imread("imageio:astronaut.png")


def _imread(uri: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("Please install imageio fetch data") from None
    return iio.imread(uri)  # type: ignore [no-any-return]


def cosem_dataset(
    uri: str = "",
    dataset: str = "jrc_hela-3",
    label: str = "er-mem_pred",
    level: int = 4,
) -> Any:
    """Load a dataset from the COSEM/OpenOrganelle project.

    Search for available options at: <https://openorganelle.janelia.org/datasets>

    Requires [tensorstore](https://pypi.org/project/tensorstore/) to be installed.

    Parameters
    ----------
    uri : str, optional
        The URI of the dataset to load. If not provided, the default URI is
        `f"{dataset}/{dataset}.n5/labels/{label}/s{level}/"`.
    dataset : str, optional
        The name of the dataset to load. Default is "jrc_hela-3".
    label : str, optional
        The label to load. Default is "er-mem_pred".
    level : int, optional
        The pyramid level to load. Default is 4.
    """
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
            # 1GB cache... but i don't think it's working
            "cache_pool": {"total_bytes_limit": 1e9},
        },
    ).result()
    ts_array = ts_array[ts.d[:].label["z", "y", "x"]]
    return ts_array[ts.d[("y", "x", "z")].transpose[:]]


def rgba() -> np.ndarray:
    """3D RGBA dataset: `(256, 256, 256, 4)`, uint8."""
    img = np.zeros((256, 256, 256, 4), dtype=np.uint8)

    # R,G,B are simple
    for i in range(256):
        img[:, i, :, 0] = i  # Red
        img[:, i, :, 2] = 255 - i  # Blue
    for j in range(256):
        img[:, :, j, 1] = j  # Green

    # Alpha is a bit trickier - requires a meshgrid for efficient computation
    x, y, z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")
    alpha = np.sqrt((x - 128) ** 2 + (y - 128) ** 2 + (z - 128) ** 2)
    img[:, :, :, 3] = np.clip(alpha, 0, 255)

    return img
