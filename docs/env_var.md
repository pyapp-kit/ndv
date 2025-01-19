# Environment Variables

`ndv` recognizes the following environment variables:

<small>*Boolean variables can be set to `1`, `0`, `True`, or `False` (case insensitive).*</small>

| <div style="width:174px">Variable</div>    | Description     | Default |
|--------------------------------------------|-----------------| ------- |
| **`NDV_CANVAS_BACKEND`** | Explicitly choose the graphics library: `"vispy"` or `"pygfx"`           | auto    |
| **`NDV_GUI_FRONTEND`**   | Explicitly choose the GUI library: `"qt"`, `"wx"`, or `"jupyter"`        | auto    |
| **`NDV_DEBUG_EXCEPTIONS`** | Whether to drop into a debugger when an exception is raised. (for development) | `False` |
| **`NDV_EXIT_ON_EXCEPTION`**  | Whether to exit the application on the first unhandled exception. (for development)  | `False` |
| **`NDV_IPYTHON_MAGIC`** | Whether to use [`%gui` magic](https://ipython.readthedocs.io/en/stable/config/eventloops.html#integrating-with-gui-event-loops) when running in IPython, to enable interactive usage. | `True`  |
| **`NDV_SYNCHRONOUS`** | Whether to force data request/draw to be synchronous. (*note: this currently has no effect on Jupyter, which is always asynchronous) | `False` |

## Framework selection

Depending on how you've [installed ndv](./install.md), you may end up with
multiple supported GUI or graphics libraries installed. You can control which
ones `ndv` uses with **`NDV_CANVAS_BACKEND`** and **`NDV_GUI_FRONTEND`**,
respectively, as described above.  Note that currently, only one GUI framework
can be used per session.

!!! info "Defaults"

    **GUI:**

    `ndv` tries to be aware of the GUI library you are using. So it will use
    `jupyter` if you are in a Jupyter notebook, `qt` if a `QApplication` is
    already running, and `wx` if a `wx.App` is already running. Finally, it
    will check for the availability of libraries in the order of `qt`, `wx`,
    `jupyter`.

    **Graphics:**

    If you have both VisPy and pygfx installed, `ndv` will (currently) default
    to using VisPy.
