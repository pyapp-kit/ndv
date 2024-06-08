# ndv

[![License](https://img.shields.io/pypi/l/ndv.svg?color=green)](https://github.com/pyapp-kit/ndv/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ndv.svg?color=green)](https://pypi.org/project/ndv)
[![Python Version](https://img.shields.io/pypi/pyversions/ndv.svg?color=green)](https://python.org)
[![CI](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml/badge.svg)](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pyapp-kit/ndv/branch/main/graph/badge.svg)](https://codecov.io/gh/pyapp-kit/ndv)

Simple, fast-loading, asynchronous, n-dimensional array viewer for Qt, with minimal dependencies.

```python
import ndv

data = ndv.data.cells3d()
# or ndv.data.nd_sine_wave()
# or *any* arraylike object (see support below)

ndv.imshow(data)
```

![Montage](https://github.com/pyapp-kit/ndv/assets/1609449/712861f7-ddcb-4ecd-9a4c-ba5f0cc1ee2c)

As an alternative to `ndv.imshow()`, you can instantiate the `ndv.NDViewer` (`QWidget` subclass) directly

```python
from qtpy.QtWidgets import QApplication
from ndv import NDViewer

app = QApplication([])
viewer = NDViewer(data)
viewer.show()
app.exec()
```

## `ndv.NDViewer`

- supports arbitrary number of dimensions, with 2D/3D view canvas, and sliders for all non-visible dims
- sliders support integer as well as slice (range)-based slicing
- colormaps provided by [cmap](https://github.com/tlambert03/cmap)
- supports [vispy](https://github.com/vispy/vispy) and [pygfx](https://github.com/pygfx/pygfx) backends
- supports any numpy-like duck arrays, including (but not limited to):
  - `numpy.ndarray`
  - `cupy.ndarray`
  - `dask.array.Array`
  - `jax.Array`
  - `pyopencl.array.Array`
  - `sparse.COO`
  - `tensorstore.TensorStore` (supports named dimensions)
  - `torch.Tensor` (supports named dimensions)
  - `xarray.DataArray` (supports named dimensions)
  - `zarr` (supports named dimensions)
- You can add support for your own storage class by subclassing `ndv.DataWrapper`
    and implementing a couple methods. (This doesn't require modifying ndv,
    but contributions of new wrappers are welcome!)

See examples for each of these array types in [examples](./examples/)

## Installation

The only required dependencies are `numpy` and `superqt[cmap,iconify]`.
You will also need a Qt backend (PyQt or PySide) and one of either
[vispy](https://github.com/vispy/vispy) or [pygfx](https://github.com/pygfx/pygfx):

```python
pip install ndv[pyqt,vispy]
```
