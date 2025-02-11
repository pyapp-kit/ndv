from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import LUTModel
from ndv.models._array_display_model import ChannelMode
from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.views import _app

if TYPE_CHECKING:
    import numpy.typing as npt

    from ndv._types import MouseMoveEvent
    from ndv.views.bases._graphics._canvas_elements import ImageHandle

    ChannelKey = int | str | None


class StreamingViewer:
    """2D streaming data viewer."""

    def __init__(self) -> None:
        # mapping of channel keys to their respective controllers
        # where None is the default channel
        self._lut_controllers: dict[ChannelKey, ChannelController] = {}
        self._handles: dict[ChannelKey, ImageHandle] = {}

        frontend_cls = _app.get_array_view_class()
        canvas_cls = _app.get_array_canvas_class()
        self._canvas = canvas_cls()
        self._canvas.mouseMoved.connect(self._on_canvas_mouse_moved)
        self._canvas.set_ndim(2)
        self._view = frontend_cls(
            self._canvas.frontend_widget(), _ArrayDataDisplayModel()
        )
        self._app = _app.ndv_app()

    def set_channel_mode(self, mode: str | ChannelMode) -> None:
        self._view.set_channel_mode(ChannelMode(mode))

    def setup(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
        channels: int | Mapping[ChannelKey, LUTModel] | None = 1,
    ) -> None:
        """Prepare the viewer for streaming data."""
        if isinstance(channels, int):
            channels = {i: LUTModel() for i in range(channels)}
        elif isinstance(channels, (Mapping, Iterable)):
            channels = {
                k: LUTModel.model_validate(v) for k, v in dict(channels).items()
            }

        else:
            raise TypeError(
                f"channels must be an int, Mapping, or Iterable, not {type(channels)}"
            )

        for key, model in channels.items():
            lut_views = [self._view.add_lut_view()]
            data = np.zeros(shape, dtype=dtype)
            self._handles[key] = handle = self._canvas.add_image(data)
            self._lut_controllers[key] = ctrl = ChannelController(
                key=key,
                lut_model=model,
                views=lut_views,
            )
            ctrl.add_handle(handle)

        info_str = f"Streaming: {len(channels)}x {shape} {dtype}"
        self._view.set_data_info(info_str)

        self._canvas.set_range()
        self._canvas.refresh()

    def update_data(
        self, data: npt.NDArray, channel: ChannelKey = 0, *, clear_others: bool = False
    ) -> None:
        """Set the data to display."""
        ctrl = self._lut_controllers[channel]
        ctrl.update_texture_data(data, direct=True)
        ctrl._auto_scale()
        if clear_others:
            for key, ctrl in self._lut_controllers.items():
                if key != channel:
                    ctrl.update_texture_data(np.zeros_like(data), direct=True)
        self._app.process_events()

    def set_lut(self, lut: LUTModel, channel: int = 0) -> None:
        """Set the LUT for a channel."""
        ...

    def show(self) -> None:
        """Show the viewer."""
        self._view.set_visible(True)

    def hide(self) -> None:
        """Hide the viewer."""
        self._view.set_visible(False)

    def close(self) -> None:
        """Close the viewer."""
        self._view.set_visible(False)

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

    def _on_canvas_mouse_moved(self, event: MouseMoveEvent) -> None:
        """Respond to a mouse move event in the view."""
        x, y, _z = self._canvas.canvas_to_world((event.x, event.y))

        # collect and format intensity values at the current mouse position
        channel_values = self._get_values_at_world_point(int(x), int(y))
        vals = []
        for ch, value in channel_values.items():
            # restrict to 2 decimal places, but remove trailing zeros
            fval = f"{value:.2f}".rstrip("0").rstrip(".")
            fch = f"{ch}: " if ch is not None else ""
            vals.append(f"{fch}{fval}")
        text = f"[{y:.0f}, {x:.0f}] " + ",".join(vals)
        self._view.set_hover_info(text)

    def _get_values_at_world_point(self, x: int, y: int) -> dict[ChannelKey, float]:
        # TODO: handle 3D data
        if x < 0 or y < 0:
            return {}

        values: dict[ChannelKey, float] = {}
        for key, ctrl in self._lut_controllers.items():
            if (value := ctrl.get_value_at_index((y, x))) is not None:
                values[key] = value

        return values
