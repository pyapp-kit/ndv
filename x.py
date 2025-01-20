from rich import print

from ndv.models._scene.nodes.node import Node

root = Node(name="A")
root.children.append(Node(name="B"))


print(root)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
json = root.model_dump_json(indent=2)
print(json)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(Node.model_validate_json(json))


# print(Node.model_json_schema())
