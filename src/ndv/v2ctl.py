from collections.abc import Container, Hashable, Mapping, Sequence
from typing import Annotated, Any, Protocol, TypeAlias

from psygnal import SignalInstance
from pydantic import BeforeValidator, Field

from .models._array_display_model import ArrayDisplayModel, AxisKey
from .models._base_model import NDVModel
from .viewer._backends._protocols import PImageHandle
from .viewer._data_wrapper import DataWrapper

DataWrapperType: TypeAlias = Annotated[DataWrapper, BeforeValidator(DataWrapper.create)]


class DataDisplayModel(NDVModel):
    """Combination of data and display models.

    Mostly this class exists to resolve AxisKeys in the display model
    (which can be axis labels, or positive/negative integers) to real/existing
    positive indices in the data.
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    data: DataWrapperType | None = None

    def _canonicalize_axis_key(self, axis: Hashable) -> int:
        """Return positive index for AxisKey (which can be +/- int or label)."""
        if self.data is None:
            raise ValueError("Data not set")

        try:
            return self.data.canonicalized_axis_map[axis]
        except KeyError as e:
            if isinstance(axis, int):
                raise IndexError(
                    f"Axis index {axis} out of bounds for data with {self.data.ndim} "
                    "dimensions"
                ) from e
            raise IndexError(f"Axis label {axis} not found in data dimensions") from e

    def canonical_data_coords(self) -> Mapping[int, Sequence]:
        """Return the coordinates of the data in canonical form."""
        if self.data is None:
            return {}
        return {
            self._canonicalize_axis_key(ax): c for ax, c in self.data.coords.items()
        }

    def canonical_visible_axes(self) -> Sequence[int]:
        """Return the visible axes in canonical form."""
        return {self._canonicalize_axis_key(ax) for ax in self.display.visible_axes}

    def current_slice(self) -> Mapping[int, int | slice]:
        """Return the current index request for the data.

        This reconciles the `current_index` and `visible_axes` attributes of the display
        with the available dimensions of the data to return a valid index request.
        In the returned mapping, the keys are the canonicalized axis indices and the
        values are either integers or slices (where axis present in `visible_axes` are
        guaranteed to be slices rather than integers).
        """
        if self.data is None:
            return {}

        requested_slice = {
            self._canonicalize_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }
        for ax in self.display.visible_axes:
            ax_ = self._canonicalize_axis_key(ax)
            if not isinstance(requested_slice.get(ax_), slice):
                requested_slice[ax_] = slice(None)
        return requested_slice

    def current_data(self) -> Any:
        """Return the data slice requested by the current index (synchronous)."""
        return self.data.isel(self.current_slice())


class PView(Protocol):
    currentIndexChanged: SignalInstance

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def add_image_to_canvas(self, data: Any) -> PImageHandle: ...
    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...


class ViewerController:
    def __init__(self, view: PView) -> None:
        self.view = view
        self._dd_model = DataDisplayModel()  # rename me
        self._set_model_connected(self._dd_model.display)
        self.view.currentIndexChanged.connect(self._on_slider_value_changed)

    # -------------- possibly move this logic up to DataDisplayModel --------------
    @property
    def model(self) -> ArrayDisplayModel:
        """Return the display model for the viewer."""
        return self._dd_model.display

    @model.setter
    def model(self, display_model: ArrayDisplayModel) -> None:
        """Set the display model for the viewer."""
        previous_model, self._dd_model.display = self._dd_model.display, display_model
        self._set_model_connected(previous_model, False)
        self._set_model_connected(display_model)

    @property
    def data(self) -> Any:
        """Return data being displayed."""
        if self._dd_model.data is None:
            return None
        return self._dd_model.data.data  # returning the actual data, not the wrapper

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        if data is None:
            self._dd_model.data = None
            return
        self._dd_model.data = DataWrapper.create(data)

        self.view.create_sliders(self._dd_model.canonical_data_coords())
        self._update_visible_sliders()
        self._update_canvas()

    # -----------------------------------------------------------------------------

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
            (model.luts.value_changed, self._on_luts_changed),
            (model.default_lut.events.cmap, self._on_luts_changed),
        ]:
            getattr(obj, _connect)(callback)

    def _update_visible_sliders(self) -> None:
        """Update which sliders are visible based on the current data and model."""
        self.view.hide_sliders(
            self._dd_model.canonical_visible_axes(), show_remainder=True
        )

    def _on_current_index_changed(self) -> None:
        value = self.model.current_index
        self.view.set_current_index(value)
        self._update_canvas()

    def _on_slider_value_changed(self) -> None:
        """Update the model when slider value changes."""
        self.model.current_index.update(self.view.current_index())

    def _update_canvas(self) -> None:
        if not self._dd_model.data:
            return

        # TODO:
        # for now we just clear and manage a single data handle
        # needs to be expanded to handle multiple textures/channels
        data = self._dd_model.current_data()  # make asynchronous
        if hdl := getattr(self, "_handle", None):
            hdl.remove()
        self._handle = self.view.add_image_to_canvas(data)
        self._handle.cmap = self.model.default_lut.cmap

    def _on_luts_changed(self) -> None:
        self._update_canvas()
