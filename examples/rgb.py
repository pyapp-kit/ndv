import ndv
from ndv.models import ArrayDisplayModel

display = ArrayDisplayModel(channel_mode="rgb")
n = ndv.imshow(ndv.data.astronaut(), display_model=display)
