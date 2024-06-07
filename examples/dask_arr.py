from __future__ import annotations

import numpy as np

try:
    from dask.array.core import map_blocks
except ImportError:
    raise ImportError("Please `pip install dask[array]` to run this example.")

frame_size = (1024, 1024)


def _dask_block(block_id: tuple[int, int, int, int, int]) -> np.ndarray | None:
    if isinstance(block_id, np.ndarray):
        return None
    data = np.random.randint(0, 255, size=frame_size, dtype=np.uint8)
    return data[(None,) * 3]


chunks = [(1,) * x for x in (1000, 64, 3)]
chunks += [(x,) for x in frame_size]
dask_arr = map_blocks(_dask_block, chunks=chunks, dtype=np.uint8)

if __name__ == "__main__":
    from qtpy import QtWidgets

    from ndv import NDViewer

    qapp = QtWidgets.QApplication([])
    v = NDViewer(dask_arr)
    v.show()
    qapp.exec()
