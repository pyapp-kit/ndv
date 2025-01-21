import numpy as np
from rich import print

from ndv import run_app
from ndv.models._scene._transform import Transform
from ndv.models._scene.nodes import Image, Points, Scene
from ndv.models._scene.view import View
from ndv.views import _app

_app.ndv_app()
img1 = Image(
    name="Some Image", data=np.random.randint(0, 255, (100, 100)).astype(np.uint8)
)

img2 = Image(
    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
    cmap="viridis",
    transform=Transform().scaled((0.7, 0.7)).translated((-10, 20)),
)
points = Points(
    coords=np.random.randint(0, 200, (100, 2)),
    size=5,
    face_color="blue",
    edge_color="yellow",
    edge_width=0.5,
    opacity=0.1,
)
scene = Scene(children=[points, img1, img2])
view = View(scene=scene)


print(view)
view.show()
view.camera._set_range(margin=0.05)
run_app()

# sys.exit()

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
json = view.model_dump_json(indent=2)
print(json)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(View.model_validate_json(json))


assert View.model_json_schema()
