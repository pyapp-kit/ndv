# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "imageio[tifffile]",
# ]
# ///
from __future__ import annotations

import os

import ndv

# os.environ["NDV_CANVAS_BACKEND"] = "pygfx"
os.environ["NDV_GUI_FRONTEND"] = "wx"

try:
    img = ndv.data.cells3d()
except Exception as e:
    print(e)
    img = ndv.data.nd_sine_wave((10, 3, 8, 512, 512))

viewer = ndv.imshow(img)
