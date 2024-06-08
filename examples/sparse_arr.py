from __future__ import annotations

try:
    import sparse
except ImportError:
    raise ImportError("Please install sparse to run this example")

import numpy as np

import ndv

shape = (256, 4, 512, 512)
N = int(np.prod(shape) * 0.001)
coords = np.random.randint(low=0, high=shape, size=(N, len(shape))).T
data = np.random.randint(0, 256, N)


# Create the sparse array from the coordinates and data
sparse_array = sparse.COO(coords, data, shape=shape)

ndv.imshow(sparse_array)
