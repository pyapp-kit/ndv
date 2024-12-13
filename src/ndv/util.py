"""Utility and convenience functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ndv.controller import ViewerController
from ndv.models import ArrayDisplayModel, DataDisplayModel
from ndv.views._app import run_app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from .models._array_display_model import ArrayDisplayModelKwargs
    from .models.data_wrappers import DataWrapper


def imshow(
    data: Any | DataWrapper, **kwargs: Unpack[ArrayDisplayModelKwargs]
) -> ViewerController:
    """Display an array or DataWrapper in a new NDViewer window.

    Parameters
    ----------
    data : Any | DataWrapper
        The data to be displayed. If not a DataWrapper, it will be wrapped in one.
    kwargs : Unpack[ArrayDisplayModelKwargs]
        Additional keyword arguments to pass to the NDViewer

    Returns
    -------
    ViewerController
        The viewer window.
    """
    display = ArrayDisplayModel.model_validate(kwargs)
    data_model = DataDisplayModel(display=display)
    data_model.data = data
    viewer = ViewerController(data_model)
    viewer.show()
    run_app()
    return viewer
