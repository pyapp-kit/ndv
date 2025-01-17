"""Utility and convenience functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ndv.controllers import ArrayViewer
from ndv.views._app import run_app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from .models._array_display_model import ArrayDisplayModel, ArrayDisplayModelKwargs
    from .models._data_wrapper import DataWrapper


@overload
def imshow(
    data: Any | DataWrapper, /, display_model: ArrayDisplayModel = ...
) -> ArrayViewer: ...
@overload
def imshow(
    data: Any | DataWrapper, /, **kwargs: Unpack[ArrayDisplayModelKwargs]
) -> ArrayViewer: ...
def imshow(
    data: Any | DataWrapper,
    /,
    display_model: ArrayDisplayModel | None = None,
    **kwargs: Unpack[ArrayDisplayModelKwargs],
) -> ArrayViewer:
    """Display an array or DataWrapper in a new NDViewer window.

    Parameters
    ----------
    data : Any | DataWrapper
        The data to be displayed. Any ArrayLike object or an `ndv.DataWrapper`.
    display_model: ArrayDisplayModel, optional
        The display model to use. If not provided, a new one will be created.
    kwargs : Unpack[ArrayDisplayModelKwargs]
        Additional keyword arguments to pass to the NDViewer

    Returns
    -------
    ArrayViewer
        The viewer window.
    """
    viewer = ArrayViewer(data, display_model, **kwargs)
    viewer.show()

    run_app()
    return viewer
