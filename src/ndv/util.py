"""Utility and convenience functions."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from qtpy.QtWidgets import QApplication

from .viewer._viewer import NDViewer

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication

    from .viewer._data_wrapper import DataWrapper


def imshow(
    data: Any | DataWrapper,
    cmap: Any | None = None,
    *,
    channel_mode: Literal["mono", "composite", "auto"] = "auto",
) -> NDViewer:
    """Display an array or DataWrapper in a new NDViewer window.

    Parameters
    ----------
    data : Any | DataWrapper
        The data to be displayed. If not a DataWrapper, it will be wrapped in one.
    cmap : Any | None, optional
        The colormap(s) to use for displaying the data.
    channel_mode : Literal['mono', 'composite'], optional
        The initial mode for displaying the channels. By default "mono" will be
        used unless a cmap is provided, in which case "composite" will be used.

    Returns
    -------
    NDViewer
        The viewer window.
    """
    app, should_exec = _get_app()
    if cmap is not None:
        channel_mode = "composite"
        if not isinstance(cmap, (list, tuple)):
            cmap = [cmap]
    elif channel_mode == "auto":
        channel_mode = "mono"
    channel_axis = None
    shape = getattr(data, "shape", [None])
    if shape[-1] in (3, 4):
        try:
            has_alpha = shape[-1] == 4
            channel_mode = "composite"
            cmap = ["red", "green", "blue"]
            data = _transpose_color(data).squeeze()
            if has_alpha:
                data = data[:3, ...]
            channel_axis = 0
        except Exception:
            warnings.warn(
                "Failed to interpret data as RGB(A), falling back to mono", stacklevel=2
            )
    viewer = NDViewer(
        data, colormaps=cmap, channel_mode=channel_mode, channel_axis=channel_axis
    )
    viewer.show()
    viewer.raise_()
    if should_exec:
        app.exec()
    return viewer


def _transpose_color(data: Any) -> Any:
    """Move the color axis to the front of the array."""
    if xr := sys.modules.get("xarray"):
        if isinstance(data, xr.DataArray):
            data = data.data
    return np.moveaxis(data, -1, 0).squeeze()


def _get_app() -> tuple[QCoreApplication, bool]:
    is_ipython = False
    if (app := QApplication.instance()) is None:
        app = QApplication([])
        app.setApplicationName("ndv")
    elif (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        is_ipython = str(shell.active_eventloop).startswith("qt")

    return app, not is_ipython
