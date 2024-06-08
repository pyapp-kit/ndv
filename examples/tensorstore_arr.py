from __future__ import annotations

try:
    import tensorstore as ts
except ImportError:
    raise ImportError("Please install tensorstore to run this example")


import ndv

data = ndv.data.cells3d()

ts_array = ts.open(
    {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "transform": {
            # tensorstore supports labeled dimensions
            "input_labels": ["z", "c", "y", "x"],
        },
    },
    create=True,
    shape=data.shape,
    dtype=data.dtype,
).result()
ts_array[:] = ndv.data.cells3d()

ndv.imshow(ts_array)
