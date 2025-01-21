from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ndv.models._scene.nodes import node as core_node

if TYPE_CHECKING:
    from pygfx.geometries import Geometry
    from pygfx.materials import Material
    from pygfx.objects import WorldObject

    from ndv.models._scene import Transform


class Node(core_node.NodeAdaptorProtocol):
    """Node adaptor for pygfx Backend."""

    _pygfx_node: WorldObject
    _material: Material
    _geometry: Geometry
    _name: str

    def _vis_get_native(self) -> Any:
        return self._pygfx_node

    def _vis_set_name(self, arg: str) -> None:
        # not sure pygfx has a name attribute...
        # TODO: for that matter... do we need a name attribute?
        # Could this be entirely managed on the model side/
        self._name = arg

    def _vis_set_parent(self, arg: core_node.Node | None) -> None:
        raise NotImplementedError

    def _vis_set_children(self, arg: list[core_node.Node]) -> None:
        # This is probably redundant with _vis_add_node
        # could maybe be a clear then add *arg
        raise NotImplementedError

    def _vis_set_visible(self, arg: bool) -> None:
        self._pygfx_node.visible = arg

    def _vis_set_opacity(self, arg: float) -> None:
        self._material.opacity = arg

    def _vis_set_order(self, arg: int) -> None:
        self._pygfx_node.render_order = arg

    def _vis_set_interactive(self, arg: bool) -> None:
        # this one requires knowledge of the controller
        raise NotImplementedError

    def _vis_set_transform(self, arg: Transform) -> None:
        self._pygfx_node.matrix = arg.root  # TODO: check this

    def _vis_add_node(self, node: core_node.Node) -> None:
        self._pygfx_node.add(node.backend_adaptor("pygfx")._vis_get_native())

    def _vis_force_update(self) -> None:
        pass

    def _vis_block_updates(self) -> None:
        pass

    def _vis_unblock_updates(self) -> None:
        pass
