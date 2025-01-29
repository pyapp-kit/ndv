# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "zarr",
#     "fsspec",
#     "aiohttp",
# ]
# ///
from __future__ import annotations

import ndv

try:
    import zarr
    import zarr.storage

    URL = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-3/jrc_hela-3.zarr/recon-1/em/fibsem-uint8"

    zarr_arr = zarr.open(URL, mode="r")
except ImportError:
    raise ImportError("Please `pip install zarr aiohttp` to run this example")


ndv.imshow(zarr_arr["s4"], current_index={1: 30}, visible_axes=(0, 2))
