from typing import Literal

from .node import Node


class Scene(Node):
    """A Root node for a scene graph.

    This really isn't anything more than a regular Node, but it's an explicit
    marker that this node is the root of a scene graph.
    """

    node_type: Literal["scene"] = "scene"
