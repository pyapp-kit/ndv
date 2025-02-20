import os

# os.environ['NDV_CANVAS_BACKEND'] = "pygfx"
os.environ["NDV_GUI_FRONTEND"] = "wx"
import ndv

n = ndv.imshow(ndv.data.rgba())
