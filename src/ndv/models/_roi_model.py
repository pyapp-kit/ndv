from __future__ import annotations

from typing import Any

from pydantic import field_validator, model_validator

from ndv.models._base_model import NDVModel


class RectangularROIModel(NDVModel):
    """Representation of an axis-aligned rectangular Region of Interest (ROI).

    Attributes
    ----------
    visible : bool
        Whether to display this roi.
    bounding_box : tuple[tuple[float, float], tuple[float, float]]
        The minimum and maximum (x, y) points of the region in data space
        (i.e. array indices, not scaled world coordinates). These two points
        define an axis-aligned bounding box.
    """

    visible: bool = True
    bounding_box: tuple[tuple[float, float], tuple[float, float]] = ((0, 0), (0, 0))

    @field_validator("bounding_box", mode="after")
    @classmethod
    def _validate_bounding_box(
        cls, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        x1 = min(bb[0][0], bb[1][0])
        y1 = min(bb[0][1], bb[1][1])
        x2 = max(bb[0][0], bb[1][0])
        y2 = max(bb[0][1], bb[1][1])
        return ((x1, y1), (x2, y2))

    @model_validator(mode="before")
    @classmethod
    def _cast_tuple(cls, values: Any) -> Any:
        if isinstance(values, tuple):
            return {"bounding_box": values}
        return values
