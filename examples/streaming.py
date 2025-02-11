import time

# unfortunate leaking of abstraction... to schedule a callback
from qtpy.QtCore import QTimer

import ndv
from ndv import StreamingViewer

viewer = StreamingViewer()
viewer.show()

shape, dtype = (1024, 1024), "uint8"

cells = ndv.data.cells3d()

viewer.setup(
    cells.shape[-2:],
    cells.dtype,
    channels={
        0: {"cmap": "green"},
        1: {"cmap": "magenta"},
    },
)


def stream(n: int = 50) -> None:
    for plane in cells:
        for c, channel in enumerate(plane):
            viewer.update_data(channel, channel=c, clear_others=(c == 0))
            time.sleep(0.05)


QTimer.singleShot(1000, stream)
ndv.run_app()
