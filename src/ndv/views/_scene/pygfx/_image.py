from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygfx

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap

    from ndv._types import ArrayLike, ImageInterpolation
    from ndv.models._scene import nodes


class Image(Node):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial

    def __init__(self, image: nodes.Image, **backend_kwargs: Any) -> None:
        if (data := image.data) is not None:
            dim = data.ndim
            if dim > 2 and data.shape[-1] <= 4:
                dim -= 1  # last array dim is probably (a subset of) rgba
        else:
            dim = 2
        # TODO: unclear whether get_view() is better here...
        self._texture = pygfx.Texture(data, dim=dim)
        self._geometry = pygfx.Geometry(grid=self._texture)
        self._material = pygfx.ImageBasicMaterial(
            clim=image.clims,
            # map=str(image.cmap), # TODO: map needs to be a TextureView
        )
        # TODO: gamma?
        # TODO: interpolation?
        self._pygfx_node = pygfx.Image(self._geometry, self._material)

    def _vis_set_cmap(self, arg: Colormap) -> None:
        self._material.map = arg

    def _vis_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _vis_set_gamma(self, arg: float) -> None:
        raise NotImplementedError

    def _vis_set_interpolation(self, arg: ImageInterpolation) -> None:
        raise NotImplementedError

    def _vis_set_data(self, arg: ArrayLike) -> None:
        raise NotImplementedError
