from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ndv.models import ArrayDisplayModel, DataWrapper
from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views import _app

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any, Unpack

    from typing_extensions import Self, TypeAlias

    from ndv._types import MouseMoveEvent
    from ndv.controllers._channel_controller import ChannelController
    from ndv.models import ChannelMode, DataWrapper
    from ndv.models._array_display_model import ArrayDisplayModelKwargs

    LutKey: TypeAlias = Hashable | None

__all__ = ["_BaseArrayViewer"]


class _BaseArrayViewer:
    """Common functionality shared between ArrayViewer and StreamingViewer."""

    def __init__(
        self,
        data: Any | DataWrapper = None,
        /,
        display_model: ArrayDisplayModel | None = None,
        **kwargs: Unpack[ArrayDisplayModelKwargs],
    ):
        if display_model is not None and kwargs:
            warnings.warn(
                "When display_model is provided, kwargs are be ignored.",
                stacklevel=2,
            )
        self._data_model = _ArrayDataDisplayModel(
            data_wrapper=data, display=display_model or ArrayDisplayModel(**kwargs)
        )
        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[LutKey, ChannelController] = {}

        # get and create the front-end and canvas classes
        self._app = _app.ndv_app()
        ArrayView = _app.get_array_view_class()
        ArrayCanvas = _app.get_array_canvas_class()

        self._viewer_model = ArrayViewerModel()

        self._canvas = ArrayCanvas(self._viewer_model)
        self._canvas.set_ndim(self._data_model.display.n_visible_axes)
        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)
        self._last_mouse_pos: tuple[float, float] | None = None

        self._view = ArrayView(
            self._canvas.frontend_widget(), self._data_model, self._viewer_model
        )
        self._view.currentIndexChanged.connect(self._on_view_current_index_changed)
        self._view.resetZoomClicked.connect(self._on_view_reset_zoom_clicked)
        self._view.channelModeChanged.connect(self._on_view_channel_mode_changed)
        self._view.visibleAxesChanged.connect(self._on_view_visible_axes_changed)

    def widget(self) -> Any:
        """Return the native front-end widget.

        !!! Warning

            If you directly manipulate the frontend widget, you're on your own :smile:.
            No guarantees can be made about synchronization with the model.  It is
            exposed for embedding in an application, and for experimentation and custom
            use cases.  Please [open an
            issue](https://github.com/pyapp-kit/ndv/issues/new) if you have questions.
        """
        return self._view.frontend_widget()

    def show(self) -> None:
        """Show the viewer."""
        self._view.set_visible(True)

    def hide(self) -> None:
        """Hide the viewer."""
        self._view.set_visible(False)

    def close(self) -> None:
        """Close the viewer."""
        self._view.set_visible(False)

    def clone(self) -> Self:
        """Return a new ArrayViewer instance with the same data and display model.

        Currently, this is a shallow copy.  Modifying one viewer will affect the state
        of the other.
        """
        # TODO: provide deep_copy option
        return type(self)(
            self._data_model.data_wrapper, display_model=self._data_model.display
        )

    def _on_view_current_index_changed(self) -> None:
        """Update the model when slider value changes."""
        self._data_model.display.current_index.update(self._view.current_index())

    def _on_view_visible_axes_changed(self) -> None:
        """Update the model when the visible axes change."""
        self._data_model.display.visible_axes = self._view.visible_axes()  # type: ignore [assignment]

    def _on_view_reset_zoom_clicked(self) -> None:
        """Reset the zoom level of the canvas."""
        self._canvas.set_range()

    def _on_view_channel_mode_changed(self, mode: ChannelMode) -> None:
        self._data_model.display.channel_mode = mode

    def _on_canvas_mouse_moved(self, event: MouseMoveEvent) -> None:
        """Respond to a mouse move event in the view."""
        x, y, _z = self._canvas.canvas_to_world((event.x, event.y))
        self._last_mouse_pos = (x, y)
        # collect and format intensity values at the current mouse position
        self._update_hover_info()

    def _update_hover_info(self) -> None:
        if not self._last_mouse_pos:
            return
        x, y = self._last_mouse_pos
        channel_values = self._get_values_at_world_point(int(x), int(y))
        vals = []
        for ch, value in channel_values.items():
            # restrict to 2 decimal places, but remove trailing zeros
            fval = f"{value:.2f}".rstrip("0").rstrip(".")
            fch = f"{ch}: " if ch is not None else ""
            vals.append(f"{fch}{fval}")
        text = f"[{y:.0f}, {x:.0f}] " + ",".join(vals)
        self._view.set_hover_info(text)

    def _get_values_at_world_point(self, x: int, y: int) -> dict[LutKey, float]:
        # TODO: handle 3D data
        if (
            x < 0 or y < 0
        ) or self._data_model.display.n_visible_axes != 2:  # pragma: no cover
            return {}

        values: dict[LutKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((y, x))) is not None:
                values[key] = value

        return values
