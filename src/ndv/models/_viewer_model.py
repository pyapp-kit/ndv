from enum import Enum, auto
from typing import TYPE_CHECKING

from ndv.models._base_model import NDVModel

if TYPE_CHECKING:
    from psygnal import Signal, SignalGroup


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
    show_3d_button : bool, optional
        Whether to show the 3D button, by default True.
    show_histogram_button : bool, optional
        Whether to show the histogram button, by default True.
    show_reset_zoom_button : bool, optional
        Whether to show the reset zoom button, by default True.
    show_roi_button : bool, optional
        Whether to show the ROI button, by default True.
    show_channel_mode_selector : bool, optional
        Whether to show the channel mode selector, by default True.
    show_progress_spinner : bool, optional
        Whether to show the progress spinner, by default
    """

    interaction_mode: InteractionMode = InteractionMode.PAN_ZOOM
    show_3d_button: bool = True
    show_histogram_button: bool = True
    show_reset_zoom_button: bool = True
    show_roi_button: bool = True
    show_channel_mode_selector: bool = True
    show_progress_spinner: bool = False

    if TYPE_CHECKING:
        # just to make IDE autocomplete better
        # it's still hard to indicate dynamic members in the events group
        class ArrayViewerModelEvents(SignalGroup):
            """Signal group for ArrayViewerModel."""

            interaction_mode = Signal(InteractionMode, InteractionMode)
            show_3d_button = Signal(bool, bool)
            show_histogram_button = Signal(bool, bool)
            show_reset_zoom_button = Signal(bool, bool)
            show_roi_button = Signal(bool, bool)
            show_channel_mode_selector = Signal(bool, bool)
            show_progress_spinner = Signal(bool, bool)

        events: ArrayViewerModelEvents = ArrayViewerModelEvents()  # type: ignore
