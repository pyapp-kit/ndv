from typing import Any

from pygfx.objects import Scene as _Scene

from ndv.models import _scene as core

from ._node import Node


class Scene(Node):
    def __init__(self, scene: core.Scene, **backend_kwargs: Any) -> None:
        self._pygfx_node = _Scene(visible=scene.visible, **backend_kwargs)
        self._pygfx_node.render_order = scene.order

        for node in scene.children:
            node.backend_adaptor()  # create backend adaptor if it doesn't exist
            self._vis_add_node(node)
