from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QSlider, QVBoxLayout, QWidget
from superqt import QLabeledSlider
from superqt.utils import signals_blocked

from .models._array_display_model import ArrayDisplayModel, AxisKey
from .viewer._backends import get_canvas_class
from .viewer._data_wrapper import DataWrapper

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from .viewer._backends import PCanvas


class DimsSliders(QWidget):
    valueChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__()
        self._sliders: MutableMapping[Hashable, QSlider] = {}

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def update_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        # TODO: consider what the axis key is here.
        # it's possible this should be canonicalized before calling this method
        # because the axis key is used to index the sliders internally
        for axis, _coords in coords.items():
            self._sliders[axis] = sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.valueChanged.connect(self.valueChanged)
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            else:
                raise NotImplementedError("Only range is supported for now")
            cast("QFormLayout", self.layout()).addRow(str(axis), sld)

    def hide_dimensions(
        self, dims: Iterable[Hashable], *, show_remainder: bool = True
    ) -> None:
        """Hide sliders corresponding to dimensions in `dims`."""
        # TODO: dims here much be the same as those used to index the sliders
        # in update_sliders.
        _dims = set(dims)
        layout = cast("QFormLayout", self.layout())
        for ax, slider in self._sliders.items():
            if ax in _dims:
                layout.setRowVisible(slider, False)
            elif show_remainder:
                layout.setRowVisible(slider, True)

    def value(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def setValue(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        for axis, val in value.items():
            if isinstance(val, slice):
                raise NotImplementedError("Slices are not supported yet")
            self._sliders[axis].setValue(val)


class Viewer(QWidget):
    _data_wrapper: DataWrapper | None
    _display_model: ArrayDisplayModel

    def __init__(
        self, data: Any = None, display_model: ArrayDisplayModel | None = None
    ):
        super().__init__()
        self.model = display_model or ArrayDisplayModel()

        self._dims_sliders = DimsSliders()
        self._dims_sliders.valueChanged.connect(self._on_sliders_value_changed)
        self._canvas: PCanvas = get_canvas_class()()

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas.qwidget())
        layout.addWidget(self._dims_sliders)

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
            return
        else:
            self._data_wrapper = DataWrapper.create(data)

        # the data is the thing that tells us how many sliders to show
        # when the data changes we get dims and coords from the data
        # TODO: we need some sort of signal from the DataWrapper to trigger update
        # short of full replacement of the data
        dims = self._data_wrapper.dims
        coords = {
            self._canonicalize_axis_key(ax, dims): c
            for ax, c in self._data_wrapper.coords.items()
        }
        self._dims_sliders.update_sliders(coords)
        self._update_visible_sliders()

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
        self._update_visible_sliders()
        self._canvas.set_ndim(self.model.n_visible_axes)

    def _update_visible_sliders(self) -> None:
        """Hide sliders corresponding to "visible" axes."""
        if self._data_wrapper is None:
            return
        dims = self._data_wrapper.dims
        hide = {self._canonicalize_axis_key(ax, dims) for ax in self.model.visible_axes}
        self._dims_sliders.hide_dimensions(hide, show_remainder=True)

    def _on_current_index_changed(self) -> None:
        value = self.model.current_index
        with signals_blocked(self._dims_sliders):
            self._dims_sliders.setValue(value)
        self._update_canvas()

    def _on_sliders_value_changed(self) -> None:
        value = self._dims_sliders.value()
        self.model.current_index.update(value)

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

    def _update_canvas(self) -> None:
        # get the data from the data wrapper
        if self._data_wrapper is None:
            return
        idx_request = self._current_index_request()
        data = self._data_wrapper.isel(idx_request)
        if hdl := getattr(self, "_handle", None):
            hdl.remove()
        self._handle = self._canvas.add_image(data)
        self._canvas.set_range()

    def _on_channel_axis_changed(self, value: AxisKey) -> None:
        print("Channel axis changed:", value)

    def _on_luts_changed(self) -> None:
        print("LUTs changed", self.model.luts)
