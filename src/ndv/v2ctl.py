from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Any, Protocol

from psygnal import SignalInstance

from .models._array_display_model import ArrayDisplayModel, AxisKey
from .viewer._backends._protocols import PImageHandle
from .viewer._data_wrapper import DataWrapper


class ViewP(Protocol):
    currentIndexChanged: SignalInstance

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def add_image_to_canvas(self, data: Any) -> PImageHandle: ...
    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...


class ViewerController:
    _data_wrapper: DataWrapper | None
    _display_model: ArrayDisplayModel

    def __init__(self, view: ViewP, model: ArrayDisplayModel | None = None) -> None:
        self.view = view
        self.model = model or ArrayDisplayModel()
        self._data_wrapper = None
        self.view.currentIndexChanged.connect(self.on_slider_value_changed)

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
            self._set_model_connected(previous_model, False)

        self._display_model = display_model
        self._set_model_connected(display_model)

    def _set_model_connected(
        self, model: ArrayDisplayModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            # (model.events.visible_axes, self._on_visible_axes_changed),
            # the current_index attribute itself is immutable
            (model.current_index.value_changed, self._on_current_index_changed),
            # (model.events.channel_axis, self._on_channel_axis_changed),
            # TODO: lut values themselves are mutable evented objects...
            # so we need to connect to their events as well
            # (model.luts.value_changed, self._on_luts_changed),
        ]:
            getattr(obj, _connect)(callback)

    def _on_current_index_changed(self) -> None:
        value = self.model.current_index
        self.view.set_current_index(value)
        self._update_canvas()

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._data_wrapper is None:
            return None
        return self._data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        if data is None:
            self._data_wrapper = None
            return
        self._data_wrapper = DataWrapper.create(data)
        dims = self._data_wrapper.dims
        coords = {
            self._canonicalize_axis_key(ax, dims): c
            for ax, c in self._data_wrapper.coords.items()
        }
        self.view.create_sliders(coords)
        self._update_visible_sliders()
        self._update_canvas()

    def on_slider_value_changed(self) -> None:
        """Update the model when slider value changes."""
        slider_values = self.view.current_index()
        self.model.current_index.update(slider_values)
        return
        self._update_canvas()

    def _update_canvas(self) -> None:
        if not self._data_wrapper:
            return
        idx_request = self._current_index_request()
        data = self._data_wrapper.isel(idx_request)
        if hdl := getattr(self, "_handle", None):
            hdl.remove()
        self._handle = self.view.add_image_to_canvas(data)

    def _current_index_request(self) -> Mapping[int, int | slice]:
        # Generate cannocalized index request
        if self._data_wrapper is None:
            return {}

        dims = self._data_wrapper.dims
        idx_request = {
            self._canonicalize_axis_key(ax, dims): v
            for ax, v in self.model.current_index.items()
        }
        for ax in self.model.visible_axes:
            ax_ = self._canonicalize_axis_key(ax, dims)
            if not isinstance(idx_request.get(ax_), slice):
                idx_request[ax_] = slice(None)
        return idx_request

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current model."""
        dims = self._data_wrapper.dims
        visible_axes = {
            self._canonicalize_axis_key(ax, dims) for ax in self.model.visible_axes
        }
        self.view.hide_sliders(visible_axes, show_remainder=True)

    def _canonicalize_axis_key(self, axis: AxisKey, dims: Sequence[Hashable]) -> int:
        """Return positive index for AxisKey (which can be +/- int or label)."""
        # TODO: improve performance by indexing ahead of time
        if isinstance(axis, int):
            ndims = len(dims)
            ax = axis if axis >= 0 else len(dims) + axis
            if ax >= ndims:
                raise IndexError(
                    f"Axis index {axis} out of bounds for data with {ndims} dimensions"
                )
            return ax
        try:
            return dims.index(axis)
        except ValueError as e:
            raise IndexError(f"Axis label {axis} not found in data dimensions") from e
