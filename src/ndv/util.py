"""Utility and convenience functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ndv.viewers import ArrayViewer
from ndv.views._app import run_app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from ndv.models import ArrayDataDisplayModel

    from .models._array_display_model import ArrayDisplayModel, ArrayDisplayModelKwargs
    from .models.data_wrappers import DataWrapper


@overload
def imshow(model: ArrayDataDisplayModel, /) -> ArrayViewer: ...
@overload
def imshow(
    data: Any | DataWrapper, display_model: ArrayDisplayModel, /
) -> ArrayViewer: ...
@overload
def imshow(
    data: Any | DataWrapper, /, **kwargs: Unpack[ArrayDisplayModelKwargs]
) -> ArrayViewer: ...
def imshow(
    data: Any | DataWrapper | ArrayDataDisplayModel,
    display_model: ArrayDisplayModel | None = None,
    /,
    **kwargs: Unpack[ArrayDisplayModelKwargs],
) -> ArrayViewer:
    """Display an array or DataWrapper in a new NDViewer window.

    Parameters
    ----------
    data : Any | DataWrapper
        The data to be displayed. If not a DataWrapper, it will be wrapped in one.
    display_model: ArrayDisplayModel, optional
        The display model to use. If not provided, a new one will be created.
    kwargs : Unpack[ArrayDisplayModelKwargs]
        Additional keyword arguments to pass to the NDViewer

    Returns
    -------
    ViewerController
        The viewer window.
    """
    viewer = ArrayViewer(data, display_model, **kwargs)  # type: ignore [arg-type]
    viewer.show()

    run_app()
    return viewer
