"""Utility and convenience functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from .viewer._viewer import NDViewer

if TYPE_CHECKING:
    from .viewer._data_wrapper import DataWrapper

    class _Executable(Protocol):
        def exec(self) -> Any: ...


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
    app, existed = _get_app()
    if cmap is not None:
        channel_mode = "composite"
        if not isinstance(cmap, (list, tuple)):
            cmap = [cmap]
    elif channel_mode == "auto":
        channel_mode = "mono"
    viewer = NDViewer(data, colormaps=cmap, channel_mode=channel_mode)
    viewer.show()
    if not existed:
        app.exec()
    return viewer


def _get_app() -> tuple[_Executable, bool]:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if not (existed := app is not None):
        app = QApplication([])

    return app, existed
