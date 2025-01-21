from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pygfx

from ndv._types import ImageInterpolation

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap

    from ndv._types import ArrayLike
    from ndv.models._scene import nodes


class Image(Node):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial
    _geometry: pygfx.Geometry

    def __init__(self, image: nodes.Image, **backend_kwargs: Any) -> None:
        self._vis_set_data(image.data)
        self._material = pygfx.ImageBasicMaterial(clim=image.clims)
        self._pygfx_node = pygfx.Image(self._geometry, self._material)

    def _vis_set_cmap(self, arg: Colormap) -> None:
        self._material.map = arg.to_pygfx()

    def _vis_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _vis_set_gamma(self, arg: float) -> None:
        warnings.warn(
            "Gamma correction not supported by pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_interpolation(self, arg: ImageInterpolation) -> None:
        if arg is ImageInterpolation.BICUBIC:
            warnings.warn(
                "Bicubic interpolation not supported by pygfx",
                RuntimeWarning,
                stacklevel=2,
            )
            arg = ImageInterpolation.LINEAR
        self._material.interpolation = arg.value

    def _create_texture(self, data: ArrayLike) -> pygfx.Texture:
        if data is not None:
            dim = data.ndim
            if dim > 2 and data.shape[-1] <= 4:
                dim -= 1  # last array dim is probably (a subset of) rgba
        else:
            dim = 2
        # TODO: unclear whether get_view() is better here...
        return pygfx.Texture(data, dim=dim)

    def _vis_set_data(self, data: ArrayLike) -> None:
        self._texture = self._create_texture(data)
        self._geometry = pygfx.Geometry(grid=self._texture)
