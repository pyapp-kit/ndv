from __future__ import annotations

try:
    import xarray as xr
except ImportError:
    raise ImportError("Please install xarray to run this example")
import ndv

da = xr.tutorial.open_dataset("air_temperature").air
ndv.imshow(da, default_lut={"cmap": "thermal"})
