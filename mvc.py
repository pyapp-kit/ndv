"""Example usage of new mvc pattern."""

from collections.abc import Sequence

from ndv import data, run_app
from ndv.controller import ViewerController

# import os
# os.environ["NDV_CANVAS_BACKEND"] = "pygfx"
viewer = ViewerController()


def foo(
    new: tuple[Sequence[float], Sequence[float]],
    old: tuple[Sequence[float], Sequence[float]],
) -> None:
    print(f"Bounding box changed from {old} to {new}")


img_data = data.cells3d()
# img_data = np.tile(img_data, (1, 1, 3, 4))
viewer.data = img_data
viewer.show()
viewer.model.current_index.update({0: 32, 1: 1})
viewer.roi.events.bounding_box.connect(foo)
viewer.roi.bounding_box = ((10, 10), (100, 100))

run_app()
