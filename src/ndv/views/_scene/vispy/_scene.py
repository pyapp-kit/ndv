from typing import Any

from vispy.scene.subscene import SubScene
from vispy.visuals.filters import Clipper

from ndv.models import _scene as core

from ._node import Node


class Scene(Node):
    _vispy_node: SubScene

    def __init__(self, scene: core.Scene, **backend_kwargs: Any) -> None:
        self._vispy_node = SubScene(**backend_kwargs)
        self._vispy_node._clipper = Clipper()
        self._vispy_node.clip_children = True

        # XXX: this logic should be moved to the model
        for node in scene.children:
            node.backend_adaptor("vispy")  # create backend adaptor if it doesn't exist
            self._vis_add_node(node)
