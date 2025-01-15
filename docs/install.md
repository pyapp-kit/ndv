# Installation

`ndv` can be used in a variety of contexts. It supports various **GUI
 frameworks**, including [PyQt](https://riverbankcomputing.com/software/pyqt),
 [PySide](https://wiki.qt.io/Qt_for_Python), [wxPython](https://wxpython.org),
 and [Jupyter Lab & Notebooks](https://jupyter.org).  It also works with
 different **graphics libraries**, including [VisPy](https://vispy.org) (for
 OpenGL) and [Pygfx](https://github.com/pygfx/pygfx) (for WebGPU).

These frameworks are *not* included directly with `ndv` and must be installed
independently. We provide a set of installation extras depending on the graphics
and GUI libraries you want to use:

<!-- logic for this table in install-table.js -->
<div id="install-table"></div>

## Framework selection

If you have multiple GUI or graphics libraries installed, you can control which
ones `ndv` uses with environment variables. The following variables are
supported:

- `NDV_CANVAS_BACKEND`: Set to `"vispy"` or `"pygfx"` to choose the graphics library.
- `NDV_GUI_FRONTEND`: Set to `"qt"`, `"wx"`, or `"jupyter"` to choose the GUI library.

!!! info "Defaults"

    **GUI:**

    `ndv` tries to be aware of the GUI library you are using. So it will use
    `jupyter` if you are in a Jupyter notebook, `qt` if a `QApplication` is
    already running, and `wx` if a `wx.App` is already running. Finally, it
    will check available libraries in the order of `qt`, `wx`, `jupyter`.

    **Graphics:**

    If you have both VisPy and pygfx installed, `ndv` will (currently) default
    to using vispy.
