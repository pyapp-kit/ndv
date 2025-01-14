"""Utility and convenience functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication

    from . import NDViewer
    from ._old_data_wrapper import DataWrapper


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
    from . import NDViewer

    app, should_exec = _get_app()
    if cmap is not None:
        channel_mode = "composite"
        if not isinstance(cmap, (list, tuple)):
            cmap = [cmap]
    elif channel_mode == "auto":
        channel_mode = "mono"
    viewer = NDViewer(data, colormaps=cmap, channel_mode=channel_mode)
    viewer.show()
    viewer.raise_()
    if should_exec:
        app.exec()
    return viewer


def _get_app() -> tuple[QCoreApplication, bool]:
    from qtpy.QtWidgets import QApplication

    is_ipython = False
    if (app := QApplication.instance()) is None:
        app = QApplication([])
        app.setApplicationName("ndv")
    elif (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        is_ipython = str(shell.active_eventloop).startswith("qt")

    return app, not is_ipython
