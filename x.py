from rich import print

from ndv.viewer_v2 import Viewer

v = Viewer()
v.model = {"luts": {1: "green"}}
v.model.luts[1] = "viridis"
print(v.model)
