import numpy as np
from qtpy.QtWidgets import QApplication
from vispy import scene

qapp = QApplication([])
canvas = scene.SceneCanvas(keys="interactive", size=(800, 600))

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Create the image
img_data = np.random.rand(512, 512).astype(np.float32)

image = scene.visuals.Image(img_data, interpolation="nearest", parent=view.scene)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.flip = (0, 1, 0)
view.camera.set_range()

canvas.show()
qapp.exec()
