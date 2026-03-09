# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "tensorstore",
# ]
# ///
from __future__ import annotations

import ndv

try:
    from ndv.data import cosem_dataset

    ts_array = cosem_dataset()
except ImportError:
    raise ImportError("Please install tensorstore to run this example")

ndv.imshow(ts_array)
