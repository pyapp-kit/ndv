from __future__ import annotations

import numpy as np
import pytest

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._pygfx._array_canvas import GfxArrayCanvas


def _force_canvas_size(canvas: GfxArrayCanvas, w: int = 600, h: int = 600) -> None:
    """Force the rendercanvas to report a valid size (needed before show)."""
    rc = canvas._canvas
    rc._size_info.set_physical_size(w, h, 1.0)


@pytest.mark.usefixtures("any_app")
def test_zoom_center() -> None:
    """Zoom should keep the center point fixed in world space."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(2)
    canvas.add_image(np.random.rand(100, 100).astype(np.float32))
    canvas.set_range()

    cam = canvas._camera
    assert cam is not None

    # Record initial state (copy position since it's mutable)
    initial_pos = tuple(cam.local.position)
    initial_zoom = cam.zoom

    # Zoom in at a specific world point
    center = (30.0, 70.0)
    canvas.zoom(factor=0.5, center=center)

    # Camera zoom should have increased (zoomed in)
    assert cam.zoom > initial_zoom

    # Camera should have moved toward the center point
    new_pos = tuple(cam.local.position)
    assert abs(center[0] - new_pos[0]) < abs(center[0] - initial_pos[0])
    assert abs(center[1] - new_pos[1]) < abs(center[1] - initial_pos[1])

    # Zoom back out by the inverse factor
    canvas.zoom(factor=2.0, center=center)

    # Should return approximately to initial state
    restored_pos = tuple(cam.local.position)
    assert restored_pos[:2] == pytest.approx(initial_pos[:2])
    assert cam.zoom == pytest.approx(initial_zoom)

    canvas.close()
