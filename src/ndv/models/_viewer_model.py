from enum import Enum, auto

from ndv.models._base_model import NDVModel


class InteractionMode(Enum):
    """An enum defining graphical interaction mechanisms with an array Viewer."""

    PAN_ZOOM = auto()  # Mode allowing the user to pan and zoom
    CREATE_ROI = auto()  # Mode where user clicks create ROIs


class ArrayViewerModel(NDVModel):
    """Representation of an array viewer.

    TODO: This will likely contain other fields including:
        * Dimensionality
        * Camera position
        * Camera frustum

    Parameters
    ----------
    interaction_mode : InteractionMode
        Describes the current interaction mode of the Viewer.
    """

    interaction_mode: InteractionMode = InteractionMode.PAN_ZOOM
