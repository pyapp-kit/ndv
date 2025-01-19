# Motivation and Scope

It can be informative to know what problems the developers were trying to solve
when creating a library, and under what constraints. `ndv` was created by a
former [napari](https://napari.org) core developer and collaborators out of a
desire to quickly view multi-dimensional arrays with minimal import times
minimal dependencies. The original need was for a component in a broader
(microscope control) application, where a fast and minimal viewer was needed to
display data.

## Goals

- [x] **N-dimensional viewer**: The current focus is on viewing multi-dimensional
    arrays (and currently just a single array at a time), with sliders
    controlling slicing from an arbitrary number of dimensions. 2D and 3D
    volumetric views are supported.
- [x] **Minimal dependencies**: `ndv` should have as few dependencies  as
    possible (both direct and indirect). Installing `napari[all]==0.5.5` into a
    clean environment brings a total of 126 dependencies. `ndv[qt]==0.2.0` has
    29, and we aim to keep that low.
- [x] **Quick to import**: `ndv` should import and show a viewer in a reasonable
    amount of time. "Reasonable" is of course relative and subjective, but we
    aim for less than 1 second on a modern laptop (currently at <100ms).
- [x] **Broad GUI Compatibility**: A common feature request for `napari` is to support
    Jupyter notebooks. `ndv` [can work with](./install.md) Qt,
    wxPython, *and* Jupyter.
- [x] **Flexible Graphics Providers**: `ndv` works with VisPy in a classical OpenGL
    context, but has an abstracting layer that allows for other graphics engines.
    We currently also support `pygfx`, a WGPU-based graphics engine.
- [x] **Model/View architecture**: `ndv` should have a clear separation between the
    data model and the view. The model should be serializable and easily
    transferable between different views. (The primary model is currently
    [`ArrayDisplayModel`][ndv.models.ArrayDisplayModel])
- [x] **Asynchronous first**: `ndv` should be asynchronous by default: meaning
    that the data request/response process happens in the background, and the
    GUI remains responsive. (Optimization of remote, multi-resolution data is on
    the roadmap, but not currently implemented).

## Scope and Roadmap

We *do* want to support the following features:

- [ ] **Multiple data sources**: We want to allow for multiple data sources to be
    displayed in the same viewer, with flexible coordinate transforms.
- [ ] **Non-image data**: We would like to support non-image data, such as points
    segmentation masks, and meshes.
- [ ] **Multi-resolution (pyramid) data**: We would like to support multi-resolution
    data, to allow for fast rendering of large datasets based on the current view.
- [ ] **Frustum culling**: We would like to support frustum culling to allow for
    efficient rendering of large datasets.
- [ ] **Ortho-viewer**: `ndv`'s clean model/view separation should allow for
    easy creation of an ortho-viewer (e.g. synchronized `XY`, `XZ`, `YZ` views).

## Non-Goals

We *do not* plan to support the following features in the near future
(if ever):

- **Oblique Slicing**: While not an explicit non-goal, oblique slicing (à la
    [Big Data Viewer](https://imagej.net/plugins/bdv/)) is different enough
    that it won't realistically be implemented in the near future.
- **Image Processing**: General image processing is out of scope. We aim to
    provide a viewer, not a full image processing library.
- **Interactive segmentation and painting**: While extensible mouse event handling
    *is* in scope, we don't intend to implement painting or  interactive
    segmentation tools.
- **Plugins**: We don't intend to support a plugin architecture. We aim to keep
    the core library as small as possible, and encourage users to build on top
    of it with their own tools.
