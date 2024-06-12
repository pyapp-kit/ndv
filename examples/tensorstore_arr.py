from __future__ import annotations

try:
    import tensorstore as ts
except ImportError:
    raise ImportError("Please install tensorstore to run this example")


import ndv

ts_array = ts.open(
    {
        "driver": "n5",
        "kvstore": {
            "driver": "s3",
            "bucket": "janelia-cosem-datasets",
            "path": "jrc_hela-3/jrc_hela-3.n5/labels/er-mem_pred/s4/",
        },
    },
).result()
ts_array = ts_array[ts.d[:].label["z", "y", "x"]]
ndv.imshow(ts_array[ts.d[("y", "x", "z")].transpose[:]])
