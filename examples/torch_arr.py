# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "torch>=2.5",
# ]
# ///
from __future__ import annotations

try:
    import torch
except ImportError:
    raise ImportError("Please install torch to run this example")

import warnings

import ndv

warnings.filterwarnings("ignore", "Named tensors")  # Named tensors are experimental

# Example usage
try:
    torch_data = torch.tensor(  # type: ignore [call-arg]
        ndv.data.nd_sine_wave(),
        names=("t", "c", "z", "y", "x"),
    )
except TypeError:
    print("Named tensors are not supported in your version of PyTorch")
    torch_data = torch.tensor(ndv.data.nd_sine_wave())

ndv.imshow(torch_data, visible_axes=("y", -1))
