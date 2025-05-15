from enum import Enum
from typing import TYPE_CHECKING

import cmap
import cmap._colormap
from pydantic import Field

from ndv.models._base_model import NDVModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypedDict

    from psygnal import Signal, SignalGroup

    class ArrayViewerModelKwargs(TypedDict, total=False):
        """Keyword arguments for `ArrayViewerModel`."""

        default_luts: "Sequence[cmap._colormap.ColorStopsLike]"
        interaction_mode: "InteractionMode"
        show_controls: bool
        show_3d_button: bool
        show_histogram_button: bool
        show_reset_zoom_button: bool
        show_roi_button: bool
        show_channel_mode_selector: bool
        show_play_button: bool
        show_data_info: bool
        show_progress_spinner: bool


class InteractionMode(str, Enum):
    """An enum defining graphical interaction mechanisms with an array Viewer."""

    PAN_ZOOM = "pan_zoom"
    CREATE_ROI = "create_roi"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


def _default_luts() -> "list[cmap.Colormap]":
    return [
        cmap.Colormap(x)
        for x in ("gray", "green", "magenta", "cyan", "red", "blue", "yellow")
    ]


class ArrayViewerModel(NDVModel):
    """Options and state for the [`ArrayViewer`][ndv.ArrayViewer].

    Attributes
    ----------
    interaction_mode : InteractionMode
        Describes the current interaction mode of the Viewer.
    show_controls : bool, optional
        Control visibility of *all* controls at once. By default True.
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
    show_play_button : bool, optional
        Whether to show the play button, by default True.
    show_progress_spinner : bool, optional
        Whether to show the progress spinner, by default
    show_data_info : bool, optional
        Whether to show shape, dtype, size, etc. about the array
    default_luts : list[cmap.Colormap], optional
        List of colormaps to use when populating the LUT dropdown menu in the viewer.
        Only editable upon initialization. Values may be any `cmap`
        [ColormapLike](https://cmap-docs.readthedocs.io/en/stable/colormaps/#colormaplike-objects)
        object (most commonly, just a string name of the colormap, like
        "gray" or "viridis").
    """

    interaction_mode: InteractionMode = InteractionMode.PAN_ZOOM
    show_controls: bool = True
    show_3d_button: bool = True
    show_histogram_button: bool = True
    show_reset_zoom_button: bool = True
    show_roi_button: bool = True
    show_channel_mode_selector: bool = True
    show_play_button: bool = True
    show_data_info: bool = True
    show_progress_spinner: bool = False
    default_luts: list[cmap.Colormap] = Field(
        default_factory=_default_luts, frozen=True
    )

    if TYPE_CHECKING:
        # just to make IDE autocomplete better
        # it's still hard to indicate dynamic members in the events group
        class ArrayViewerModelEvents(SignalGroup):
            """Signal group for ArrayViewerModel."""

            interaction_mode = Signal(InteractionMode, InteractionMode)
            show_controls = Signal(bool, bool)
            show_3d_button = Signal(bool, bool)
            show_histogram_button = Signal(bool, bool)
            show_reset_zoom_button = Signal(bool, bool)
            show_roi_button = Signal(bool, bool)
            show_channel_mode_selector = Signal(bool, bool)
            show_play_button = Signal(bool, bool)
            show_data_info = Signal(bool, bool)
            show_progress_spinner = Signal(bool, bool)

        events: ArrayViewerModelEvents = ArrayViewerModelEvents()  # type: ignore
