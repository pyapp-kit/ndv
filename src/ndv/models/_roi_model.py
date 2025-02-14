from __future__ import annotations

from pydantic import field_validator

from ndv.models._base_model import NDVModel


class RectangularROIModel(NDVModel):
    """Representation of an axis-aligned rectangular Region of Interest (ROI).

    Attributes
    ----------
    visible : bool
        Whether to display this roi.
    bounding_box : tuple[tuple[float, float], tuple[float, float]]
        The minimum (2D) point and the maximum (2D) point contained within the
        region. Using these two points, an axis-aligned bounding box can be
        constructed.
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
