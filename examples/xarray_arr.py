from __future__ import annotations

import xarray as xr
from qtpy import QtWidgets

from ndv import NDViewer

da = xr.tutorial.open_dataset("air_temperature").air

if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    v = NDViewer(da, colormaps=["thermal"], channel_mode="composite")
    v.show()
    qapp.exec()
