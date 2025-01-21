from contextlib import suppress

import numpy as np
from rich import print

from ndv import run_app
from ndv.models._scene import Transform
from ndv.models._scene.nodes import Image, Points, Scene
from ndv.models._scene.view import View
from ndv.views import _app

_app.ndv_app()
img1 = Image(
    name="Some Image",
    data=np.random.randint(0, 255, (100, 100)).astype(np.uint8),
    clims=(0, 255),
)

img2 = Image(
    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
    cmap="viridis",
    transform=Transform().scaled((0.7, 0.5)).translated((-10, 20)),
    clims=(0, 255),
)

scene = Scene(children=[img1, img2])
with suppress(Exception):
    points = Points(
        coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
        size=5,
        face_color="coral",
        edge_color="blue",
        opacity=0.8,
    )
    scene.children.insert(0, points)
view = View(scene=scene)


print(view)
view.show()
view.camera._set_range(margin=0.05)
run_app()

# sys.exit()

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
json = view.model_dump_json(indent=2, exclude_unset=True)
print(json)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
obj = View.model_validate_json(json)
print(obj)


assert View.model_json_schema()
