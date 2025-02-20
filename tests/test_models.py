from unittest.mock import Mock

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._roi_model import RectangularROIModel


def test_array_display_model() -> None:
    m = ArrayDisplayModel()

    mock = Mock()
    m.events.channel_axis.connect(mock)
    m.current_index.item_added.connect(mock)
    m.current_index.item_changed.connect(mock)

    m.channel_axis = 4
    mock.assert_called_once_with(4, None)  # new, old
    mock.reset_mock()
    m.current_index["5"] = 1
    mock.assert_called_once_with(5, 1)  # key, value
    mock.reset_mock()
    m.current_index[5] = 4
    mock.assert_called_once_with(5, 4, 1)  # key, new, old
    mock.reset_mock()

    assert ArrayDisplayModel.model_json_schema(mode="validation")
    assert ArrayDisplayModel.model_json_schema(mode="serialization")


def test_rectangular_roi_model() -> None:
    m = RectangularROIModel()

    mock = Mock()
    m.events.bounding_box.connect(mock)
    m.events.visible.connect(mock)

    m.bounding_box = ((10, 10), (20, 20))
    mock.assert_called_once_with(
        ((10, 10), (20, 20)),  # New bounding box value
        ((0, 0), (0, 0)),  # Initial bounding box on construction
    )
    mock.reset_mock()

    m.visible = False
    mock.assert_called_once_with(
        False,  # New visibility
        True,  # Initial visibility on construction
    )
    mock.reset_mock()

    assert RectangularROIModel.model_json_schema(mode="validation")
    assert RectangularROIModel.model_json_schema(mode="serialization")
