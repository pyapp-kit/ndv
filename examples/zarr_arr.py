from __future__ import annotations

import ndv

try:
    import zarr
    import zarr.storage
except ImportError:
    raise ImportError("Please `pip install zarr aiohttp` to run this example")


URL = "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tczyx.ome.zarr"
zarr_arr = zarr.open(URL, mode="r")

ndv.imshow(zarr_arr["s0"])
