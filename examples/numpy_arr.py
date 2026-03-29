from __future__ import annotations

import ndv

try:
    img = ndv.data.cells3d()
except Exception as e:
    print(e)
    img = ndv.data.nd_sine_wave((10, 3, 8, 512, 512))

viewer = ndv.imshow(
    img,
    current_index={0: 30},
    channel_mode="composite",
    luts={0: {"name": "FITC"}, 1: {"name": "DAPI", "cmap": "magenta"}},
    scales={0: 0.4, -1: 0.2, -2: 0.2},
    viewer_options={"use_shared_histogram": True},
)
