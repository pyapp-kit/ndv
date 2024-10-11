import numpy as np
from qtpy.QtWidgets import QApplication
from vispy.scene import SceneCanvas

from ndv.histogram._vispy import PlotWidget

app = QApplication([])
pw = PlotWidget()
canvas = SceneCanvas()
canvas.central_widget.add_widget(pw)
canvas.show()

pw.histogram(np.random.normal(10, 10, 10000), bins=100, color="red")
pw.histogram(np.random.normal(-10, 4, 10000), bins=120, color=(0, 0, 1, 0.5))
pw.lock_axis("y")
app.exec()
