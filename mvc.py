"""Example usage of new mvc pattern."""

from collections.abc import Sequence

import numpy as np
from qtpy.QtWidgets import QApplication

from ndv import data
from ndv.controller import ViewerController

app = QApplication([])

viewer = ViewerController()  # ultimately, this will be the public api

# viewer.data = data.cosem_dataset(level=5)
viewer.data = np.tile(data.cells3d(), (2, 1, 3, 4))
viewer._view.show()


def foo(
    new: tuple[Sequence[float], Sequence[float]],
    old: tuple[Sequence[float], Sequence[float]],
) -> None:
    print(f"Bounding box changed from {old} to {new}")


viewer.roi.events.bounding_box.connect(foo)
viewer.roi.bounding_box = ([10, 10], [100, 100])
app.exec()
