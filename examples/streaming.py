import time

# unfortunate leaking of abstraction... to schedule a callback
from qtpy.QtCore import QTimer

import ndv
from ndv import StreamingViewer

viewer = StreamingViewer()
viewer.show()

cells = ndv.data.cells3d()
viewer.setup(
    cells.shape[-2:],
    cells.dtype,
    channels={
        0: {"cmap": "green"},
        1: {"cmap": {"name": "indigo", "value": ["#000", "#AF22FF"]}},
    },
)


def stream(n: int = 50) -> None:
    for plane in cells:
        for c, channel in enumerate(plane):
            viewer.update_data(channel, channel=c, clear_others=(c == 0))
            time.sleep(0.01)


QTimer.singleShot(500, stream)
ndv.run_app()
