# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "xarray",
#     "scipy",
#     "pooch",
# ]
# ///
from __future__ import annotations

try:
    import xarray as xr
except ImportError:
    raise ImportError("Please install xarray[io] to run this example")
import ndv

da = xr.tutorial.open_dataset("air_temperature").air
ndv.imshow(da, default_lut={"cmap": "thermal"}, visible_axes=("lat", "lon"))
