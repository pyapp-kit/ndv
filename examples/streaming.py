# /// script
# dependencies = [
#     "ndv[pyqt,pygfx]",
#     "imageio[tifffile]",
# ]
# ///
import time

import ndv
from ndv import StreamingViewer

viewer = StreamingViewer()
viewer.show()

cells = ndv.data.cells3d()
viewer.setup(
    cells.shape[-2:],
    cells.dtype,
    channels={0: {"cmap": "green"}, 1: {"cmap": "magenta", "clims": (1500, 21000)}},
)


def stream() -> None:
    for plane in cells:
        for c, channel in enumerate(plane):
            viewer.update_data(channel, channel=c, clear_others=(c == 0))
            time.sleep(0.01)
            # bit of a hack to force updates for this example
            viewer._app.process_events()


ndv.call_later(200, stream)
ndv.run_app()
