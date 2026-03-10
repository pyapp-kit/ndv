from __future__ import annotations

import numpy as np
import pytest

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._vispy._array_canvas import VispyArrayCanvas


@pytest.mark.usefixtures("any_app")
def test_zoom_center() -> None:
    """Zoom should keep the center point fixed in world space."""
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(2)
    canvas.add_image(np.random.rand(100, 100).astype(np.float32))
    canvas.set_range()

    cam = canvas._camera
    initial_rect = cam.rect

    # Zoom in at a specific world point
    center = (30.0, 70.0)
    canvas.zoom(factor=0.5, center=center)

    # Camera rect should have changed (smaller = zoomed in)
    new_rect = cam.rect
    assert new_rect.width < initial_rect.width

    # The center point should still be inside the rect
    assert new_rect.left <= center[0] <= new_rect.right
    assert new_rect.bottom <= center[1] <= new_rect.top

    # Zoom back out by the inverse factor
    canvas.zoom(factor=2.0, center=center)

    # Should return approximately to initial state
    restored_rect = cam.rect
    assert restored_rect.width == pytest.approx(initial_rect.width)
    assert restored_rect.height == pytest.approx(initial_rect.height)
    assert restored_rect.left == pytest.approx(initial_rect.left)
    assert restored_rect.bottom == pytest.approx(initial_rect.bottom)

    canvas.close()
