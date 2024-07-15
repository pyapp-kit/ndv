from rich import print

from ndv.model.model import ArrayDisplayModel

m = ArrayDisplayModel()
m.channel_axis_changed.connect(print)
m.current_index.item_added.connect(print)
m.current_index.item_changed.connect(print)
m.channel_axis = 4
m.current_index["5"] = 1

print(m)
