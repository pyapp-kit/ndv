from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from ndv.controllers import ArrayViewer
from ndv.controllers._channel_controller import ChannelController
from ndv.models import LUTModel
from ndv.views import _app
from ndv.views.bases._array_view import ArrayViewOptions

if TYPE_CHECKING:
    import numpy.typing as npt

    from ndv.views.bases._graphics._canvas_elements import ImageHandle

    ChannelKey = int | str | None


class StreamingViewer(ArrayViewer):
    """2D streaming data viewer."""

    def __init__(self) -> None:
        # mapping of channel keys to their respective controllers
        # where None is the default channel
        super().__init__()
        self._handles: dict[ChannelKey, ImageHandle] = {}
        self._app = _app.ndv_app()
        self._view.set_options(
            ArrayViewOptions(
                show_3d_button=False,
                show_channel_mode_selector=False,
                show_histogram_button=False,
            )
        )

    def setup(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
        channels: int | Mapping[ChannelKey, LUTModel] | None = 1,
    ) -> None:
        """Prepare the viewer for streaming data."""
        self._clear_canvas()
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
