import time
import numpy as np
import ndv

ndv.DataWrapper
img = np.zeros((100, 256, 256), dtype=np.uint8)

viewer = ndv.ArrayViewer(img)
viewer.show()


class Streamer:
    def __init__(self, data: np.ndarray, viewer: ndv.ArrayViewer):
        self.data = data
        self.viewer = viewer

    def add_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        self.data[frame_idx] = frame
        self.viewer.display_model.current_index.update({0: frame_idx})


streamer = Streamer(img, viewer)


def stream() -> None:
    for i in range(100):
        data = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        streamer.add_frame(i, data)
        time.sleep(0.05)
        viewer._app.process_events()


ndv.call_later(200, stream)
ndv.run_app()
