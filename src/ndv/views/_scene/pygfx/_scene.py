from typing import Any

import pygfx

from ndv.models import _scene as core

from ._node import Node


class Scene(Node):
    _pygfx_node: pygfx.Scene

    def __init__(self, scene: core.Scene, **backend_kwargs: Any) -> None:
        self._pygfx_node = pygfx.Scene(visible=scene.visible, **backend_kwargs)
        self._pygfx_node.render_order = scene.order

        # Almar does this in Display.show...
        self._pygfx_node.add(pygfx.AmbientLight())

        for node in scene.children:
            self._vis_add_node(node)
