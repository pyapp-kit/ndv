from collections.abc import Sequence
from typing import Self

from pydantic import model_validator

from ndv.models._base_model import NDVModel


class ROIModel(NDVModel):
    """Representation of how to display a region of interest (ROI).
    For now,
    # TODO: Consider additional parameters for non-rectangle ROIs.

    Parameters
    ----------
    visible : bool
        Whether to display this roi.
    bounding_box: tuple[Sequence[float], Sequence[float]]
        The minimum point and the maximum point contained within the region.
        Using these two points, an axis-aligned bounding box can be constructed.
    """

    visible: bool = True
    bounding_box: tuple[Sequence[float], Sequence[float]] = ([0, 0], [0, 0])

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        mi, ma = self.bounding_box
        if len(mi) != len(ma):
            raise ValueError(
                "Minimum and maximum do not share the same number of dimensions"
            )
        for i in range(len(mi)):
            if mi[i] > ma[i]:
                # TODO: Could we switch min and max?
                raise ValueError(f"Minimum is greater than maximum at index {i}")

        return self
