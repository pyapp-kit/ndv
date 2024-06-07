from __future__ import annotations

import numpy as np
import tensorstore as ts
from qtpy import QtWidgets

from ndv import NDViewer

shape = (10, 4, 3, 512, 512)
ts_array = ts.open(
    {"driver": "zarr", "kvstore": {"driver": "memory"}},
    create=True,
    shape=shape,
    dtype=ts.uint8,
).result()
ts_array[:] = np.random.randint(0, 255, size=shape, dtype=np.uint8)
ts_array = ts_array[ts.d[:].label["t", "c", "z", "y", "x"]]

if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    v = NDViewer(ts_array)
    v.show()
    qapp.exec()
