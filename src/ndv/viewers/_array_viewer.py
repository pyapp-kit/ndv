"""Utility and convenience functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, overload

from ndv.controller import ViewerController
from ndv.models import ArrayDataDisplayModel

if TYPE_CHECKING:
    from typing import Any, Unpack

    from ndv.models._array_display_model import (
        ArrayDisplayModel,
        ArrayDisplayModelKwargs,
    )
    from ndv.models.data_wrappers import DataWrapper
    from ndv.views.bases._array_view import ArrayView


class ArrayViewer:
    """Viewer dedicated to displaying a single n-dimensional array.

    This wraps a model, view, and controller into a single object, and defines the
    public API.

    Parameters
    ----------
    data_or_model : ArrayDataDisplayModel | DataWrapper | Any
        Data to be displayed. If a full `ArrayDataDisplayModel` is provided, it will be
        used directly. If an array or `DataWrapper` is provided, a default display model
        will be created.
    display_model : ArrayDisplayModel, optional
        Just the display model to use. If provided, `data_or_model` must be an array
        or `DataWrapper`... and kwargs will be ignored.
    **kwargs: ArrayDisplayModelKwargs
        Keyword arguments to pass to the `ArrayDisplayModel` constructor. If
        `display_model` is provided, these will be ignored.
    """

    @overload
    def __init__(self, model: ArrayDataDisplayModel, /) -> None: ...
    @overload
    def __init__(
        self, data: Any | DataWrapper, display_model: ArrayDisplayModel | None = ..., /
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: Any | DataWrapper,
        display_model: None = ...,
        /,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ) -> None: ...
    def __init__(
        self,
        data: Any | DataWrapper | ArrayDataDisplayModel,
        display_model: ArrayDisplayModel | None = None,
        /,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ) -> None:
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

        self._controller = ViewerController(data_model)

    @property
    def view(self) -> ArrayView:
        """Return the front-end view object."""
        return self._controller._view

    @property
    def model(self) -> ArrayDataDisplayModel:
        """Return the display model for the viewer."""
        return self._controller._model

    @property
    def data(self) -> Any | None:
        """Return the data wrapper object."""
        return self._controller.data

    def show(self) -> None:
        self.view.set_visible(True)

    def close(self) -> None:
        self.view.close()

    def add_histogram(self) -> None:
        """Add histogram to the view."""
        self._controller.add_histogram()
