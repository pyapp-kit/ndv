import sys
from collections.abc import Container, Hashable, Mapping, MutableMapping, Sequence
from typing import Any, Protocol

import cmap
from psygnal import Signal, SignalInstance
from pydantic import Field

from .models._array_display_model import ArrayDisplayModel, AxisKey
from .models._base_model import NDVModel
from .viewer._backends._protocols import PImageHandle
from .viewer._data_wrapper import DataWrapper


class DataDisplayModel(NDVModel):
    """Combination of data and display models.

    Mostly this class exists to resolve AxisKeys in the display model
    (which can be axis labels, or positive/negative integers) to real/existing
    positive indices in the data.
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    data: DataWrapper | None = None

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

    @property
    def canonical_data_coords(self) -> MutableMapping[int, Sequence]:
        """Return the coordinates of the data in canonical form."""
        if self.data is None:
            return {}
        return {
            self._canonicalize_axis_key(ax): c for ax, c in self.data.coords.items()
        }

    @property
    def canonical_visible_axes(self) -> tuple[int, ...]:
        """Return the visible axes in canonical form."""
        return tuple(
            self._canonicalize_axis_key(ax) for ax in self.display.visible_axes
        )

    @property
    def canonical_current_index(self) -> MutableMapping[int, int | slice]:
        """Return the current index in canonical form."""
        return {
            self._canonicalize_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }

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

        requested_slice = self.canonical_current_index
        for ax in self.canonical_visible_axes:
            if not isinstance(requested_slice.get(ax), slice):
                requested_slice[ax] = slice(None)

        # ensure that all axes are slices, so that we don't lose any dimensions
        # data will be squeezed to remove singleton dimensions later after
        # transposing according to the order of visible axes
        for ax, val in requested_slice.items():
            if isinstance(val, int):
                requested_slice[ax] = slice(val, val + 1)
        return requested_slice

    def current_data(self) -> Any:
        """Return the data slice requested by the current index (synchronous)."""
        data = self.data.isel(self.current_slice())  # same shape, with singleton dims
        # rearrange according to the order of visible axes
        t_dims = self.canonical_visible_axes
        t_dims += tuple(i for i in range(data.ndim) if i not in t_dims)
        return data.transpose(*t_dims).squeeze()


class PLutView(Protocol):
    visibleChanged: Signal
    autoscaleChanged: Signal
    cmapChanged: Signal
    climsChanged: Signal

    def setName(self, name: str) -> None: ...
    def setAutoScale(self, auto: bool) -> None: ...
    def setColormap(self, cmap: cmap.Colormap) -> None: ...
    def setClims(self, clims: tuple[float, float]) -> None: ...
    def setLutVisible(self, visible: bool) -> None: ...


class PView(Protocol):
    """Protocol for the view in the viewer."""

    currentIndexChanged: SignalInstance

    def refresh(self) -> None: ...
    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def add_image_to_canvas(self, data: Any) -> PImageHandle: ...
    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...

    def add_lut_view(self) -> PLutView: ...
    def show(self) -> None: ...


class ViewerController:
    """The controller mostly manages the connection between the model and the view."""

    def __init__(self, view: PView | None = None) -> None:
        if view is None:
            view = _pick_view_backend()
        self.view = view
        self._dd_model = DataDisplayModel()  # rename me
        self._set_model_connected(self._dd_model.display)
        self.view.currentIndexChanged.connect(self._on_slider_value_changed)
        self._handles: dict[int | None, PImageHandle] = {}

        self._lut_views: dict[int | None, PLutView] = {}
        self.add_lut_view(None)

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

        self.view.create_sliders(self._dd_model.canonical_data_coords)
        self._update_visible_sliders()
        self._update_canvas()

    # -----------------------------------------------------------------------------

    def _set_model_connected(
        self, model: ArrayDisplayModel, connect: bool = True
    ) -> None:
        """Connect or disconnect the model to/from the viewer.

        We do this in a single method so that we are sure to connect and disconnect
        the same events in the same order.  (but it's kinda ugly)
        """
        _connect = "connect" if connect else "disconnect"

        for obj, callback in [
            (model.events.visible_axes, self._on_visible_axes_changed),
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
            self._dd_model.canonical_visible_axes, show_remainder=True
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
        data = self._dd_model.current_data()  # TODO: make asynchronous
        if None in self._handles:
            self._handles[None].data = data
        else:
            self._handles[None] = self.view.add_image_to_canvas(data)
            self._handles[None].cmap = self.model.default_lut.cmap
            if clims := self.model.default_lut.clims:
                self._handles[None].clim = clims
        self.view.refresh()

    def _on_luts_changed(self) -> None:
        self._update_canvas()

    def _on_visible_axes_changed(self) -> None:
        self._update_visible_sliders()
        self._update_canvas()

    def add_lut_view(self, key: int | None) -> PLutView:
        if key in self._lut_views:
            # need to clean up
            raise NotImplementedError(f"LUT view with key {key} already exists")
        self._lut_views[key] = lut = self.view.add_lut_view()

        lut.visibleChanged.connect(self._on_lut_visible_changed)
        lut.autoscaleChanged.connect(self._on_autoscale_changed)
        lut.cmapChanged.connect(self._on_cmap_changed)
        lut.climsChanged.connect(self._on_clims_changed)

        model_lut = self._dd_model.display.default_lut
        model_lut.events.cmap.connect(lut.setColormap)
        model_lut.events.clims.connect(lut.setClims)
        model_lut.events.autoscale.connect(lut.setAutoScale)
        model_lut.events.visible.connect(lut.setLutVisible)
        return lut

    def _on_lut_visible_changed(self, visible: bool) -> None:
        self._handles[None].visible = visible

    def _on_autoscale_changed(self, autoscale: bool) -> None: ...

    def _on_cmap_changed(self, cmap: cmap.Colormap) -> None:
        self._handles[None].cmap = cmap

    def _on_clims_changed(self, clims: tuple[float, float]) -> None:
        self._handles[None].clim = clims


def _pick_view_backend() -> PView:
    if _is_running_in_notebook():
        from .v2view_jupyter import JupyterViewerView

        return JupyterViewerView()
    elif _is_running_in_qapp():
        from .v2view_qt import QViewerView

        return QViewerView()

    raise RuntimeError("Could not determine the appropriate viewer backend")


def _is_running_in_notebook() -> bool:
    if IPython := sys.modules.get("IPython"):
        if shell := IPython.get_ipython():
            return bool(shell.__class__.__name__ == "ZMQInteractiveShell")
    return False


def _is_running_in_qapp() -> bool:
    for mod_name in ("PyQt5", "PySide2", "PySide6", "PyQt6"):
        if mod := sys.modules.get(f"{mod_name}.QtWidgets"):
            if qapp := getattr(mod, "QApplication", None):
                return qapp.instance() is not None
    return False
