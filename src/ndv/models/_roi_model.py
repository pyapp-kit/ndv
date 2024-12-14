from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import field_validator

from ndv.models._base_model import NDVModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class BoundingBoxModel(NDVModel):
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

    @field_validator("bounding_box")
    @classmethod
    def _validate_bounding_box(cls, bb: Any) -> tuple[Sequence[float], Sequence[float]]:
        if not isinstance(bb, tuple):
            raise ValueError(f"{bb} not a tuple of points!")
        x1 = min(bb[0][0], bb[1][0])
        y1 = min(bb[0][1], bb[1][1])
        x2 = max(bb[0][0], bb[1][0])
        y2 = max(bb[0][1], bb[1][1])
        return ((x1, y1), (x2, y2))

    # @model_validator(mode="after")
    # def _validate_model(self) -> Self:
    #     mi, ma = self.bounding_box
    #     if len(mi) != len(ma):
    #         raise ValueError(
    #             "Minimum and maximum do not share the same number of dimensions"
    #         )
    #     for i in range(len(mi)):
    #         if mi[i] > ma[i]:
    #             # TODO: Could we switch min and max?
    #             raise ValueError(f"Minimum is greater than maximum at index {i}")

    #     return self
