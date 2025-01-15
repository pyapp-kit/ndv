# ndv

[![License](https://img.shields.io/pypi/l/ndv.svg?color=green)](https://github.com/pyapp-kit/ndv/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ndv.svg?color=green)](https://pypi.org/project/ndv)
[![Python Version](https://img.shields.io/pypi/pyversions/ndv.svg?color=green)](https://python.org)
[![CI](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml/badge.svg)](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pyapp-kit/ndv/branch/main/graph/badge.svg)](https://codecov.io/gh/pyapp-kit/ndv)

Simple, fast-loading, asynchronous, n-dimensional array viewer, with minimal dependencies.

```python
import ndv

data = ndv.data.cells3d()  # or any arraylike object
ndv.imshow(data)
```

![Montage](https://github.com/pyapp-kit/ndv/assets/1609449/712861f7-ddcb-4ecd-9a4c-ba5f0cc1ee2c)

[`ndv.imshow()`](https://pyapp-kit.github.io/ndv/dev/reference/ndv/#ndv.imshow)
creates an instance of
[`ndv.ArrayViewer`](https://pyapp-kit.github.io/ndv/dev/reference/ndv/controllers/#ndv.controllers.ArrayViewer),
which you can also use directly:

```python
import ndv

viewer = ndv.ArrayViewer(data)
viewer.show()
ndv.run_app()
```

> [!TIP]
> To embed the viewer in a broader Qt or wxPython application, you can
> access the viewer's `widget` attribute and add it to your layout.

## Features

- ⚡️ fast to import, fast to show
- 🪶 minimal dependencies
- 📦 supports arbitrary number of dimensions
- 🥉 2D/3D view canvas
- 🌠 supports [VisPy](https://github.com/vispy/vispy) or
  [pygfx](https://github.com/pygfx/pygfx) backends
- 🛠️ support [Qt](https://doc.qt.io), [wx](https://www.wxpython.org), or
  [Jupyter](https://jupyter.org) GUI frontends
- 🎨 colormaps provided by [cmap](https://cmap-docs.readthedocs.io/)
- 🏷️ supports named dimensions and categorical coordinate values (WIP)
- 🦆 supports most array types, including:
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

See examples for each of these array types in
[examples](https://github.com/pyapp-kit/ndv/tree/main/examples)

> [!NOTE]
> *You can add support for any custom storage class by subclassing
> `ndv.DataWrapper` and [implementing a couple
> methods](https://github.com/pyapp-kit/ndv/blob/main/examples/custom_store.py).
> (This doesn't require modifying ndv, but contributions of new wrappers are
> welcome!)*

## Installation

Because ndv supports many combinations of GUI and graphics frameworks,
you must install it along with additional dependencies for your desired backend.

See the [installation guide](https://pyapp-kit.github.io/ndv/dev/install/) for
complete details.

To just get started quickly using Qt and vispy:
  
```python
pip install ndv[qt]
```

For Jupyter with vispy, (no Qt or wxPython):

```python
pip install ndv[jup]
```

## Documentation

For more information, and complete API reference, see the
[documentation](https://pyapp-kit.github.io/ndv/).

## Motivation

This package arose from the need for a way to *quickly* view multi-dimensional
arrays with zero tolerance for long import times and/or excessive dependency
lists. I want something that I can use to view any of the many multi-dimensional
array types, out of the box, with no assumptions about dimensionality. I want it
to work reasonably well with remote, asynchronously loaded data. I also want it
to take advantage of things like named dimensions and categorical coordinate
values when available. For now, it's a Qt-only widget, since that's where the
need arose, but I can imagine a jupyter widget in the future (likely as a remote
frame buffer for vispy/pygfx).

I do not intend for this to grow into full-fledged application, or wrap a
complete scene graph, though point and ROI selection would be welcome additions.
