import ndv
from ndv.models import ArrayDisplayModel

display = ArrayDisplayModel(channel_mode="rgb")
n = ndv.imshow(ndv.data.rgba(), display_model=display)
