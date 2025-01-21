from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy import scene

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap

    from ndv._types import ArrayLike, ImageInterpolation
    from ndv.models._scene import nodes


class Image(Node):
    """Vispy backend adaptor for an Image node."""

    _vispy_node: scene.Image

    def __init__(self, image: nodes.Image, **backend_kwargs: Any) -> None:
        backend_kwargs.update(
            {
                "cmap": image.cmap.to_vispy(),
                # "clim": image.clim_applied(),
                "gamma": image.gamma,
                "interpolation": image.interpolation.value,
            }
        )
        try:
            backend_kwargs.setdefault("texture_format", "auto")
            self._vispy_node = scene.Image(image.data, **backend_kwargs)
        except Exception:
            backend_kwargs.pop("texture_format")
            self._vispy_node = scene.Image(image.data, **backend_kwargs)

    def _vis_set_cmap(self, arg: Colormap) -> None:
        self._vispy_node.cmap = arg.to_vispy()

    def _vis_set_clims(self, arg: tuple[float, float] | None) -> None:
        if arg is not None:
            self._vispy_node.clim = arg

    def _vis_set_gamma(self, arg: float) -> None:
        self._vispy_node.gamma = arg

    def _vis_set_interpolation(self, arg: ImageInterpolation) -> None:
        self._vispy_node.interpolation = arg.value

    def _vis_set_data(self, arg: ArrayLike) -> None:
        self._vispy_node.set_data(arg)
        self._vispy_node.update()
