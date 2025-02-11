from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import LUTModel
from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.views import _app

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

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
        self._canvas.set_ndim(2)
        self._view = frontend_cls(
            self._canvas.frontend_widget(), _ArrayDataDisplayModel()
        )

    def setup(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
        num_channels: int | None = None,
        channels: Mapping[ChannelKey, LUTModel] | None = None,
    ) -> None:
        """Prepare the viewer for streaming data."""
        if channels is None:
            if num_channels is None:
                num_channels = 1
            channels = {i: LUTModel() for i in range(num_channels)}

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

        self._canvas.set_range()

    def set_data(self, data: npt.NDArray, channel: ChannelKey = 0) -> None:
        """Set the data to display."""
        self._lut_controllers[channel].update_texture_data(data, direct=True)

    def set_lut(self, lut: LUTModel, channel: int = 0) -> None: ...

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
