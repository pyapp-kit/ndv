from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np

from ndv import NDViewer

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def make_lazy_array(shape: tuple[int, ...]) -> da.Array:
    rest_shape = shape[:-2]
    frame_shape = shape[-2:]

    def _dask_block(block_id: tuple[int, int, int, int, int]) -> np.ndarray | None:
        if isinstance(block_id, np.ndarray):
            return None
        size = (1,) * len(rest_shape) + frame_shape
        return np.random.randint(0, 255, size=size, dtype=np.uint8)

    chunks = [(1,) * x for x in rest_shape] + [(x,) for x in frame_shape]
    return da.map_blocks(_dask_block, chunks=chunks, dtype=np.uint8)  # type: ignore


def test_stack_viewer2(qtbot: QtBot) -> None:
    dask_arr = make_lazy_array((1000, 64, 3, 256, 256))
    v = NDViewer(dask_arr)
    qtbot.addWidget(v)
    v.show()

    # wait until there are no running jobs, because the callbacks
    # in the futures hold a strong reference to the viewer
    qtbot.waitUntil(lambda: v._last_future is None, timeout=1000)
