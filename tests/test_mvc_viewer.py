from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from ndv.controller import ArrayViewer
from ndv.views import gui_frontend


@pytest.mark.usefixtures("any_app")
def test_mvc_viewer() -> None:
    """Example usage of new mvc pattern."""
    viewer = ArrayViewer()
    assert gui_frontend() in type(viewer._view).__name__.lower()
    viewer.show()

    data = np.random.randint(0, 255, size=(10, 10, 10, 10, 10), dtype="uint8")
    viewer.data = data

    # test changing current index via the view
    index_mock = Mock()
    viewer.model.display.current_index.value_changed.connect(index_mock)
    index = {0: 4, 1: 1, 2: 2}
    # setting the index should trigger the signal, only once
    viewer._view.set_current_index(index)
    index_mock.assert_called_once()
    for k, v in index.items():
        assert viewer.model.display.current_index[k] == v
    # setting again should not trigger the signal
    index_mock.reset_mock()
    viewer._view.set_current_index(index)
    index_mock.assert_not_called()
