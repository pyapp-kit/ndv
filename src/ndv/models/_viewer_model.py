from enum import Enum, auto

from ndv.models._base_model import NDVModel


class CanvasMode(Enum):
    PAN_ZOOM = auto()
    CREATE_ROI = auto()


class ViewerModel(NDVModel):
    """Representation of a data viewer.

    Parameters
    ----------
    mode : CanvasMode
        TODO: Description
    """

    mode: CanvasMode = CanvasMode.PAN_ZOOM
