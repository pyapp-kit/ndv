import numpy as np

import ndv
from ndv._chunking import Slicer

data = np.random.rand(10, 3, 8, 5, 128, 128)
wrapper = ndv.DataWrapper.create(data)
slicer = Slicer(wrapper, chunks=(5, 1, 2, 2, 64, 34))

index = {0: 2, 1: 2, 2: 0, 3: 4}
idx = wrapper.to_conventional(index)
print(idx)
print(wrapper[idx].shape)

slicer.request_index(index)
# slicer.shutdown()
