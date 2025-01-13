from unittest.mock import Mock

from ndv.models._array_display_model import ArrayDisplayModel


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
