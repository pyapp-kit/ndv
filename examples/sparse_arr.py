from __future__ import annotations

try:
    import sparse
except ImportError:
    raise ImportError("Please install sparse to run this example")

import numpy as np

shape = (256, 4, 512, 512)
N = int(np.prod(shape) * 0.001)
coords = np.random.randint(low=0, high=shape, size=(N, len(shape))).T
data = np.random.randint(0, 256, N)


# Create the sparse array from the coordinates and data
sparse_array = sparse.COO(coords, data, shape=shape)


if __name__ == "__main__":
    from qtpy import QtWidgets

    from ndv import NDViewer

    qapp = QtWidgets.QApplication([])
    v = NDViewer(sparse_array, channel_axis=1)
    v.show()
    qapp.exec()
