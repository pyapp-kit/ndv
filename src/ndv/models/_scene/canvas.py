from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from psygnal.containers import EventedList

from ._vis_model import Field, SupportsVisibility, VisModel
from .view import View

if TYPE_CHECKING:
    import numpy as np
    from cmap import Color


ViewType = TypeVar("ViewType", bound=View)


class CanvasAdaptorProtocol(SupportsVisibility["Canvas"], Protocol):
    """Protocol defining the interface for a Canvas adaptor."""

    @abstractmethod
    def _vis_set_width(self, arg: int) -> None: ...
    @abstractmethod
    def _vis_set_height(self, arg: int) -> None: ...
    @abstractmethod
    def _vis_set_background_color(self, arg: Color | None) -> None: ...
    @abstractmethod
    def _vis_set_title(self, arg: str) -> None: ...
    @abstractmethod
    def _vis_close(self) -> None: ...
    @abstractmethod
    def _vis_render(self) -> np.ndarray: ...
    @abstractmethod
    def _vis_add_view(self, view: View) -> None: ...
    def _vis_get_ipython_mimebundle(
        self, *args: Any, **kwargs: Any
    ) -> dict | tuple[dict, dict] | Any:
        return NotImplemented


class ViewList(EventedList[ViewType]):
    def _pre_insert(self, value: ViewType) -> ViewType:
        if not isinstance(value, View):  # pragma: no cover
            raise TypeError("Canvas views must be View objects")
        return super()._pre_insert(value)


class Canvas(VisModel[CanvasAdaptorProtocol]):
    """Canvas onto which views are rendered.

    In desktop applications, this will be a window. In web applications, this will be a
    div.  The canvas has one or more views, which are rendered onto it.  For example,
    an orthoviewer might be a single canvas with three views, one for each axis.
    """

    width: float = Field(default=500, description="The width of the canvas in pixels.")
    height: float = Field(
        default=500, description="The height of the canvas in pixels."
    )
    background_color: Color | None = Field(
        default=None,
        description="The background color. None implies transparent "
        "(which is usually black)",
    )
    visible: bool = Field(default=False, description="Whether the canvas is visible.")
    title: str = Field(default="", description="The title of the canvas.")
    views: ViewList[View] = Field(default_factory=lambda: ViewList(), frozen=True)

    @property
    def size(self) -> tuple[float, float]:
        """Return the size of the canvas."""
        return self.width, self.height

    @size.setter
    def size(self, value: tuple[float, float]) -> None:
        """Set the size of the canvas."""
        self.width, self.height = value

    def close(self, backend: str | None = None) -> None:
        """Close the canvas."""
        if self.has_backend_adaptor(backend=backend):
            for adaptor in self.backend_adaptors:
                adaptor._vis_close()

    # show and render will trigger a backend connection

    def show(self, *, backend: str | None = None) -> None:
        """Show the canvas.

        Parameters
        ----------
        backend : str, optional
            The backend to use.  If not provided, the default backend will be used.
            TODO: clarify how this is chosen.
        """
        # Note: the canvas.show() method is THE primary place where we create a tree
        # of backend objects. (None of the lower level Node objects actually *need*
        # any backend representation until they need to be shown visually)
        # So, this method really bootstraps the entire "hydration" of the backend tree.
        # Here, we make sure that all of the views have a backend adaptor.

        # If you need to add any additional logic to handle the moment of backend
        # creation in a specific Node subtype, you can override the `_create_backend`
        # method (see, for example, the View._create_backend method)
        self.backend_adaptor(backend=backend)  # make sure we have a backend adaptor
        for view in self.views:
            if not view.has_backend_adaptor():
                # make sure all of the views have a backend adaptor
                view.backend_adaptor(backend=backend)
        self.visible = True

    def hide(self) -> None:
        """Hide the canvas."""
        self.visible = False

    def render(self, backend: str | None = None) -> np.ndarray:
        """Render canvas to offscren buffer and return as numpy array."""
        # TODO: do we need to set visible=True temporarily here?
        return self.backend_adaptor(backend=backend)._vis_render()

    # consider using canvas.views.append?
    def add_view(self, view: View | None = None, **kwargs: Any) -> View:
        """Add a new view to the canvas."""
        # TODO: change kwargs to params
        if view is None:
            view = View(**kwargs)
        elif kwargs:  # pragma: no cover
            warnings.warn("kwargs ignored when view is provided", stacklevel=2)
        elif not isinstance(view, View):  # pragma: no cover
            raise TypeError("view must be an instance of View")

        self.views.append(view)
        if self.has_backend_adaptor():
            for adaptor in self.backend_adaptors:
                adaptor._vis_add_view(view)
        return view

    def _repr_mimebundle_(self, *args: Any, **kwargs: Any) -> dict[str, Any] | Any:
        """Return a mimebundle for the canvas.

        This defers to the native object's _vis_get_ipython_mimebundle method
        if it exists.
        Allowing different backends to support Jupyter or other rich display.

        https://ipython.readthedocs.io/en/stable/config/integrating.html#more-powerful-methods
        """
        adaptor = self.backend_adaptor()
        if hasattr(adaptor, "_vis_get_ipython_mimebundle"):
            return adaptor._vis_get_ipython_mimebundle(*args, **kwargs)
        return NotImplemented


class GridCanvas(Canvas):
    """Subclass with numpy-style indexing."""

    # def __getitem__(self, key: tuple[int, int]) -> View:
    #     """Get the View at the given row and column."""
