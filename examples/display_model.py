from rich import print

from ndv.models._array_display_model import ArrayDisplayModel

m = ArrayDisplayModel()
print(m)

m.events.channel_axis.connect(lambda x: print(f"channel_axis: {x}"))
m.current_index.item_added.connect(lambda x, y: print(f"current_index[{x}] = {y}"))
m.current_index.item_changed.connect(
    lambda x, y, z: print(f"current_index[{x!r}] = {y} -> {z}")
)

m.channel_axis = 4
m.current_index["5"] = 1
m.current_index[5] = 4


assert ArrayDisplayModel.model_json_schema(mode="validation")
assert ArrayDisplayModel.model_json_schema(mode="serialization")
