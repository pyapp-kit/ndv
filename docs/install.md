# Installation

!!! tip "TLDR;"

    === "For a desktop usage"
 
        ```python
        pip install "ndv[vispy,pyqt]"
        ```

    === "For Jupyter notebook/lab"
    
        ```python
        pip install "ndv[vispy,jupyter]"
        ```

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

!!! note "Framework Selection"

    If you have *multiple* supported GUI or graphics libraries installed, you can
    select which ones `ndv` uses with
    [environment variables](env_var.md#framework-selection).
