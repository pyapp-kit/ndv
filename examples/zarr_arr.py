from __future__ import annotations

import zarr
import zarr.storage
from qtpy import QtWidgets

from ndv import NDViewer

URL = "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tczyx.ome.zarr"
zarr_arr = zarr.open(URL, mode="r")

if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    v = NDViewer(zarr_arr["s0"])
    v.show()
    qapp.exec()
