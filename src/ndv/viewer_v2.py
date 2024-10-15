from typing import Any

from .models._array_display_model import ArrayDisplayModel, AxisKey
from .viewer._data_wrapper import DataWrapper


class Viewer:
    def __init__(
        self, data: Any = None, display_model: ArrayDisplayModel | None = None
    ):
        self.model = display_model or ArrayDisplayModel()
        self._data_wrapper: DataWrapper | None = None
        self.data = data

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._data_wrapper is None:
            return None
        return self._data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Return the data to be displayed."""
        if data is None:
            self._data_wrapper = None
        else:
            self._data_wrapper = DataWrapper.create(data)

    @property
    def model(self) -> ArrayDisplayModel:
        """Return the display model for the viewer."""
        return self._display_model

    @model.setter
    def model(self, display_model: ArrayDisplayModel) -> None:
        """Set the display model for the viewer."""
        display_model = ArrayDisplayModel.model_validate(display_model)
        previous_model: ArrayDisplayModel | None = getattr(self, "_display_model", None)
        if previous_model is not None:
            self._set_connected(previous_model, False)

        self._display_model = display_model
        self._set_connected(display_model)

    def _set_connected(self, model: ArrayDisplayModel, connect: bool = True) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.visible_axes, self._on_visible_axes_changed),
            # the current_index attribute itself is immutable
            (model.current_index.value_changed, self._on_current_index_changed),
            (model.events.channel_axis, self._on_channel_axis_changed),
            # TODO: lut values themselves are mutable evented objects...
            # so we need to connect to their events as well
            (model.luts.value_changed, self._on_luts_changed),
        ]:
            getattr(obj, _connect)(callback)

    def _on_visible_axes_changed(
        self, value: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey]
    ) -> None:
        print("Visible axes changed:", value)

    def _on_current_index_changed(self) -> None:
        value = self.model.current_index
        # the entire evented mutable object is being changed
        print("Current index changed:", value)

    def _on_channel_axis_changed(self, value: AxisKey) -> None:
        print("Channel axis changed:", value)

    def _on_luts_changed(self) -> None:
        print("LUTs changed", self.model.luts)
