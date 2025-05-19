"""Utility and convenience functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ndv.controllers import ArrayViewer
from ndv.views._app import run_app

if TYPE_CHECKING:
    from typing import Any, Unpack

    from .models._array_display_model import ArrayDisplayModel, ArrayDisplayModelKwargs
    from .models._data_wrapper import DataWrapper
    from .models._viewer_model import ArrayViewerModel, ArrayViewerModelKwargs


@overload
def imshow(
    data: Any | DataWrapper,
    /,
    *,
    viewer_options: ArrayViewerModel | ArrayViewerModelKwargs | None = ...,
    display_model: ArrayDisplayModel = ...,
) -> ArrayViewer: ...
@overload
def imshow(
    data: Any | DataWrapper,
    /,
    *,
    viewer_options: ArrayViewerModel | ArrayViewerModelKwargs | None = ...,
    **display_kwargs: Unpack[ArrayDisplayModelKwargs],
) -> ArrayViewer: ...
def imshow(
    data: Any | DataWrapper,
    /,
    *,
    viewer_options: ArrayViewerModel | ArrayViewerModelKwargs | None = None,
    display_model: ArrayDisplayModel | None = None,
    **display_kwargs: Unpack[ArrayDisplayModelKwargs],
) -> ArrayViewer:
    """Display an array or DataWrapper in a new `ArrayViewer` window.

    This convenience function creates an `ArrayViewer` instance populated with `data`,
    calls `show()` on it, and then runs the application.

    Parameters
    ----------
    data : Any | DataWrapper
        The data to be displayed. Any ArrayLike object or an `ndv.DataWrapper`.
    display_model: ArrayDisplayModel, optional
        The display model to use. If not provided, a new one will be created.
    viewer_options: ArrayViewerModel | ArrayViewerModelKwargs, optional
        Either a [`ArrayViewerModel`][ndv.models.ArrayViewerModel] or a dictionary of
        keyword arguments used to create one.
        See docs for [`ArrayViewerModel`][ndv.models.ArrayViewerModel] for options.
    **display_kwargs : Unpack[ArrayDisplayModelKwargs]
        Additional keyword arguments used to create the
        [`ArrayDisplayModel`][ndv.models.ArrayDisplayModel]. (Generally, this is
        used instead of passing a `display_model` directly.)

    Returns
    -------
    ArrayViewer
        The `ArrayViewer` instance.
    """
    viewer = ArrayViewer(
        data,
        display_model=display_model,
        viewer_options=viewer_options,
        **display_kwargs,
    )
    viewer.show()

    run_app()
    return viewer
