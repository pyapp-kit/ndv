# ndv

[![License](https://img.shields.io/pypi/l/ndv.svg?color=green)](https://github.com/tlambert03/ndv/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ndv.svg?color=green)](https://pypi.org/project/ndv)
[![Python Version](https://img.shields.io/pypi/pyversions/ndv.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/ndv/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/ndv/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/ndv/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/ndv)

Simple, fast-loading, asynchronous, n-dimensional viewer for Qt, with minimal dependencies.

```python
from qtpy import QtWidgets
from ndv import NDViewer
from skimage import data  # just for example data here

qapp = QtWidgets.QApplication([])
v = NDViewer(data.cells3d())
v.show()
qapp.exec()
```

![Montage](https://github.com/tlambert03/ndv/assets/1609449/712861f7-ddcb-4ecd-9a4c-ba5f0cc1ee2c)

## `NDViewer`

- supports arbitrary number of dimensions, with 2D/3D view canvas, and sliders for all non-visible dims
- sliders support integer as well as slice (range)-based slicing
- colormaps provided by [cmap](https://github.com/tlambert03/cmap)
- supports [vispy](https://github.com/vispy/vispy) and [pygfx](https://github.com/pygfx/pygfx) backends
- supports any numpy-like duck arrays, with special support for features in:
  - `xarray.DataArray`
  - `dask.array.Array`
  - `tensorstore.TensorStore`
  - `zarr`
  - `dask`

## Installation

The only required dependencies are `numpy` and `superqt[cmap,iconify]`.
You will also need a Qt backend (PyQt or PySide) and one of either
[vispy](https://github.com/vispy/vispy) or [pygfx](https://github.com/pygfx/pygfx):

```python
pip install ndv[pyqt,vispy]
```
