import numpy as np
import pygfx
from qtpy.QtWidgets import QApplication
from wgpu.gui.qt import QWgpuCanvas

app = QApplication([])
canvas = QWgpuCanvas(size=(512, 512))
renderer = pygfx.renderers.WgpuRenderer(canvas)

# create data
data = np.random.rand(512, 512).astype(np.float32)

# create scene and camera
scene = pygfx.Scene()
camera = pygfx.OrthographicCamera(*data.shape)
camera.local.position = data.shape[0] / 2, data.shape[1] / 2, 0
controller = pygfx.PanZoomController(camera, register_events=renderer)

# add image
scene.add(
    pygfx.Image(
        pygfx.Geometry(grid=pygfx.Texture(data, dim=2)),
        pygfx.ImageBasicMaterial(depth_test=False),
    )
)

canvas.request_draw(lambda: renderer.render(scene, camera))
app.exec()
