from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol

from cmap import Color
from pydantic import Field

from .node import Node, NodeAdaptorProtocol

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PointsBackend(NodeAdaptorProtocol["Points"], Protocol):
    """Protocol for a backend Image adaptor object."""

    @abstractmethod
    def _vis_set_coords(self, coords: NDArray) -> None: ...
    @abstractmethod
    def _vis_set_size(self, size: float) -> None: ...
    @abstractmethod
    def _vis_set_face_color(self, face_color: Color) -> None: ...
    @abstractmethod
    def _vis_set_edge_color(self, edge_color: Color) -> None: ...
    @abstractmethod
    def _vis_set_edge_width(self, edge_width: float) -> None: ...
    @abstractmethod
    def _vis_set_symbol(self, symbol: str) -> None: ...
    @abstractmethod
    def _vis_set_scaling(self, scaling: str) -> None: ...
    @abstractmethod
    def _vis_set_antialias(self, antialias: float) -> None: ...
    @abstractmethod
    def _vis_set_opacity(self, opacity: float) -> None: ...


SymbolName = Literal[
    "disc",
    "arrow",
    "ring",
    "clobber",
    "square",
    "x",
    "diamond",
    "vbar",
    "hbar",
    "cross",
    "tailed_arrow",
    "triangle_up",
    "triangle_down",
    "star",
    "cross_lines",
]
ScalingMode = Literal[True, False, "fixed", "scene", "visual"]


class Points(Node[PointsBackend]):
    """Points that can be placed in scene."""

    node_type: Literal["points"] = "points"

    # numpy array of 2D/3D point centers, shape (N, 2) or (N, 3)
    coords: Any = Field(default=None, repr=False, exclude=True)
    size: float = Field(default=10.0, description="The size of the points.")
    face_color: Color | None = Field(
        default=Color("white"), description="The color of the faces."
    )
    edge_color: Color | None = Field(
        default=Color("black"), description="The color of the edges."
    )
    edge_width: float | None = Field(default=1.0, description="The width of the edges.")
    symbol: SymbolName = Field(
        default="disc", description="The symbol to use for the points."
    )
    # TODO: these are vispy-specific names.  Determine more general names
    scaling: ScalingMode = Field(
        default=True, description="Determines how points scale when zooming."
    )

    antialias: float = Field(default=1, description="Anti-aliasing factor, in px.")
    opacity: float = Field(default=1.0, description="The opacity of the points.")
