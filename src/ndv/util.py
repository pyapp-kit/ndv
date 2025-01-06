"""Utility and convenience functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, overload

from ndv.controller import ViewerController
from ndv.models import ArrayDataDisplayModel
from ndv.views._app import run_app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from .models._array_display_model import ArrayDisplayModel, ArrayDisplayModelKwargs
    from .models.data_wrappers import DataWrapper


@overload
def imshow(model: ArrayDataDisplayModel, /) -> ViewerController: ...
@overload
def imshow(
    data: Any | DataWrapper, display_model: ArrayDisplayModel, /
) -> ViewerController: ...
@overload
def imshow(
    data: Any | DataWrapper, /, **kwargs: Unpack[ArrayDisplayModelKwargs]
) -> ViewerController: ...
def imshow(
    data: Any | DataWrapper | ArrayDataDisplayModel,
    display_model: ArrayDisplayModel | None = None,
    /,
    **kwargs: Unpack[ArrayDisplayModelKwargs],
) -> ViewerController:
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
    if isinstance(data, ArrayDataDisplayModel):
        data_model = data
    elif display_model is not None:
        if kwargs:  # pragma: no cover
            warnings.warn(
                "Ignoring keyword arguments when display_model is provided.",
                UserWarning,
                stacklevel=2,
            )
        data_model = ArrayDataDisplayModel(display=display_model, data_wrapper=data)
    else:
        data_model = ArrayDataDisplayModel(display=kwargs, data_wrapper=data)

    viewer = ViewerController(data_model)
    viewer.show()

    run_app()
    return viewer
