from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy import scene
from vispy.visuals.transforms import MatrixTransform, NullTransform

from ndv.models._scene.nodes import node as core_node

if TYPE_CHECKING:
    from ndv.models._scene import Transform


class Node(core_node.NodeAdaptorProtocol):
    """Node adaptor for Vispy Backend."""

    _vispy_node: scene.VisualNode
    _update: Any = None

    def _vis_force_update(self) -> None:
        self._vispy_node.update()

    def _vis_block_updates(self) -> None:
        self._update, self._vispy_node.update = self._vispy_node.update, lambda: None

    def _vis_unblock_updates(self) -> None:
        self._vispy_node.update, self._update = self._update, None

    def _vis_get_native(self) -> Any:
        return self._vispy_node

    def _vis_set_name(self, arg: str) -> None:
        self._vispy_node.name = arg

    def _vis_set_parent(self, arg: core_node.Node | None) -> None:
        if arg is None:
            self._vispy_node.parent = None
        else:
            return  # this causes recursion error  # FIXME
            vispy_node = arg.backend_adaptor("vispy")._vis_get_native()
            if not isinstance(vispy_node, scene.Node):
                raise TypeError("Parent must be a Node")
            self._vispy_node.parent = vispy_node

    def _vis_set_children(self, arg: list[core_node.Node]) -> None:
        pass

    def _vis_set_visible(self, arg: bool) -> None:
        self._vispy_node.visible = arg

    def _vis_set_opacity(self, arg: float) -> None:
        self._vispy_node.opacity = arg

    def _vis_set_order(self, arg: int) -> None:
        self._vispy_node.order = arg

    def _vis_set_interactive(self, arg: bool) -> None:
        self._vispy_node.interactive = arg

    def _vis_set_transform(self, arg: Transform) -> None:
        T = NullTransform() if arg.is_null() else MatrixTransform(arg.root)
        self._vispy_node.transform = T

    def _vis_add_node(self, node: core_node.Node) -> None:
        vispy_node = node.backend_adaptor("vispy")._vis_get_native()
        if not isinstance(vispy_node, scene.Node):
            raise TypeError("Node must be a Vispy Node")
        vispy_node.parent = self._vispy_node
