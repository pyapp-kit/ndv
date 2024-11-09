from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from typing import Any

from pydantic import Field

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._base_model import NDVModel
from ndv.models._data_wrapper import DataWrapper


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

    def current_data_slice(self) -> Any:
        """Return the data slice requested by the current index (synchronous)."""
        if self.data is None:
            raise ValueError("Data not set")

        data = self.data.isel(self.current_slice())  # same shape, with singleton dims
        # rearrange according to the order of visible axes
        t_dims = self.canonical_visible_axes
        t_dims += tuple(i for i in range(data.ndim) if i not in t_dims)
        return data.transpose(*t_dims).squeeze()
