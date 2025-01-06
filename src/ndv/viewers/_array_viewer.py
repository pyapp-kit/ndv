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


class ArrayViewer:
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

    def show(self) -> None:
        self._controller.show()
