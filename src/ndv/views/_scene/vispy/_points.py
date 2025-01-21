from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy import scene

from ._node import Node

if TYPE_CHECKING:
    import numpy.typing as npt
    from cmap import Color

    from ndv.models._scene import nodes


class Points(Node):
    """Vispy backend adaptor for an Points node."""

    _vispy_node: scene.Markers

    def __init__(self, points: nodes.Points, **backend_kwargs: Any) -> None:
        backend_kwargs.update(
            {
                "scaling": points.scaling,
                "alpha": points.opacity,
                "antialias": points.antialias,
                "pos": points.coords,
                "size": points.size,
                "edge_width": points.edge_width,
                "face_color": points.face_color,
                "edge_color": points.edge_color,
                "symbol": points.symbol,
            }
        )
        self._vispy_node = scene.Markers(**backend_kwargs)

    # TODO:
    # vispy has an odd way of selectively setting individual markers properties
    # without altering the rest of the properties (you generally have to include
    # most of the state each time or you will overwrite the rest of the state)
    # this goes for size, face/edge color, edge width, symbol
    def _vis_set_coords(self, coords: npt.NDArray) -> None:
        if self._vispy_node._data is None:
            self._vispy_node.set_data(coords)

    def _vis_set_size(self, size: float) -> None:
        self._vispy_node.set_data(size=size)

    def _vis_set_face_color(self, face_color: Color) -> None:
        self._vispy_node.set_data(face_color=face_color)

    def _vis_set_edge_color(self, edge_color: Color) -> None:
        self._vispy_node.set_data(edge_color=edge_color)

    def _vis_set_edge_width(self, edge_width: float) -> None:
        return
        self._vispy_node.set_data(edge_width=edge_width)

    def _vis_set_symbol(self, symbol: str) -> None:
        self._vispy_node.symbol = symbol

    def _vis_set_scaling(self, scaling: str) -> None:
        self._vispy_node.scaling = scaling

    def _vis_set_antialias(self, antialias: float) -> None:
        self._vispy_node.antialias = antialias

    def _vis_set_opacity(self, opacity: float) -> None:
        self._vispy_node.alpha = opacity
