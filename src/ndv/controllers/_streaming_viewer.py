from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.controllers._channel_controller import ChannelController
from ndv.models import LUTModel
from ndv.views import _app
from ndv.views.bases import CanvasElement

from ._base_array_viewer import _BaseArrayViewer

if TYPE_CHECKING:
    import numpy.typing as npt

    from ndv._types import CovariantMapping
    from ndv.views.bases._graphics._canvas_elements import ImageHandle

    ChannelKey = Any


class StreamingViewer(_BaseArrayViewer):
    """2D streaming data viewer.

    Simplified viewer for streaming 2D data. This viewer is designed to display
    2D data in real-time, with minimal configuration. It (currently) has no history,
    it simply shows the last frame of data for each channel, as fast as possible.

    The main two methods are `setup` and `update_data`. The former is used to
    initialize the viewer with the shape and data type of the data to display,
    as well as the channels to display. The latter is used to update the data
    for a given channel.
    """

    def __init__(self) -> None:
        super().__init__()
        self._app = _app.ndv_app()
        self._handles: dict[ChannelKey, ImageHandle] = {}
        self._shape: tuple[int, int] | None = None
        self._dtype: np.dtype | None = None
        self._viewer_model.show_3d_button = False
        self._viewer_model.show_histogram_button = False
        self._viewer_model.show_channel_mode_selector = False

    def reset(self) -> None:
        """Reset the viewer to its initial state."""
        for lut_ctrl in self._lut_controllers.values():
            for lut_view in lut_ctrl.lut_views:
                # FIXME:
                # basically, we never want to ask the front-end ArrayView object
                # to remove graphics elements (even if they are LutViews)
                # https://github.com/pyapp-kit/ndv/issues/138
                # I'd prefer to have an `IS isinstance()` rather than not...
                # but LutView is already a low-level ABC
                if not isinstance(lut_view, CanvasElement):
                    self._view.remove_lut_view(lut_view)
            while lut_ctrl.handles:
                lut_ctrl.handles.pop().remove()
        self._handles.clear()
        self._shape = None
        self._dtype = None

    @property
    def shape(self) -> tuple[int, int] | None:
        """Return the shape that the viewer is prepared to receive.

        Call `setup` to set the shape and dtype.  (May be used to determine whether
        setup has been called.)
        """
        return self._shape

    @property
    def dtype(self) -> np.dtype | None:
        """Return the data type that the viewer is prepared to receive.

        Call `setup` to set the shape and dtype. (May be used to determine whether
        setup has been called.)
        """
        return self._dtype

    def setup(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike,
        channels: int
        | Sequence[ChannelKey]
        | CovariantMapping[ChannelKey, LUTModel | Any]
        | Iterable[tuple[ChannelKey, LUTModel | Any]]
        | None = None,
    ) -> None:
        """Prepare the viewer for streaming data.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the data to display.
        dtype : numpy.dtype
            The data type of the data to display.
        channels : int, dict, or iterable of tuples
            The channels to display. Can be one of:
                - `int`: The number of channels to display. Each channel will
                    be assigned a default `LUTModel`, and channel keys will be integers
                    starting from 0.
                - `Sequence`: A sequence of channel keys. Each key will be
                    assigned a default `LUTModel`.
                - `dict`: A dictionary mapping channel keys to `LUTModel`s.
                - `iterable of tuples`: An iterable of `(channel key, LUTModel)` pairs.
            "channel keys" are any hashable and are used to identify channels
            when updating data. If a single channel is used, the channel key can
            be omitted.
            Defaults to `None`, in which case a single channel with a default `LUTModel`
            is created.
        """
        self.reset()
        channels = self._norm_channels(channels)
        self._shape = shape
        self._dtype = np.dtype(dtype)
        for key, model in channels.items():
            lut_views = [self._view.add_lut_view(key)]
            data = np.zeros(shape, dtype=dtype)
            self._handles[key] = handle = self._canvas.add_image(data)
            self._lut_controllers[key] = ctrl = ChannelController(
                key=key,
                lut_model=model,
                views=lut_views,
            )
            ctrl.add_handle(handle)

        info_str = f"Streaming: {shape} {dtype} "
        if (nchannels := len(channels)) > 1:
            info_str += f" ({nchannels} channels)"
        self._view.set_data_info(info_str)

        self._canvas.set_range()
        self._canvas.refresh()

    def update_data(
        self,
        data: npt.NDArray,
        channel: ChannelKey = None,
        *,
        clear_others: bool = False,
    ) -> None:
        """Set the data to display.

        Data *must* be the same shape and dtype as was used to the last call to `setup`.

        Parameters
        ----------
        data : numpy.ndarray
            The data to display.
        channel : hashable, optional
            The channel to update. If `None`, the first channel is updated.
        clear_others : bool, default: False
            If `True`, all other channels are cleared. This is useful when
            displaying a sequence of multi-channel data, and you want to indicate that
            a new frame has started (and clear the previous frame).
        """
        try:
            ctrl = self._lut_controllers[channel]
        except KeyError:
            keys = list(self._lut_controllers)
            if channel is None:
                ctrl = self._lut_controllers[keys[0]]
            else:
                raise KeyError(
                    f"Channel {channel!r} not recognized. Must be one of {keys}"
                ) from None

        ctrl.update_texture_data(data, direct=True)
        mi, ma = ctrl.lut_model.clims.calc_clims(data)
        for view in ctrl.lut_views:
            view.set_clims((mi, ma))

        if clear_others:
            for key, ctrl in self._lut_controllers.items():
                if key != channel:
                    ctrl.update_texture_data(np.zeros_like(data), direct=True)
        self._update_hover_info()

    def _norm_channels(
        self,
        channels: int
        | Sequence[ChannelKey]
        | CovariantMapping[ChannelKey, LUTModel | Any]
        | Iterable[tuple[ChannelKey, LUTModel | Any]]
        | None,
    ) -> dict[ChannelKey, LUTModel]:
        if isinstance(channels, int):
            return {i: LUTModel() for i in range(channels)}
        elif not channels:
            return {None: LUTModel()}
        elif isinstance(channels, Mapping):
            return {k: LUTModel.model_validate(v) for k, v in channels.items()}
        elif isinstance(channels, Sequence):
            if not isinstance(channels, (str, bytes)):
                # Check if it's a sequence of individual keys
                return {k: LUTModel() for k in channels}
        elif isinstance(channels, Iterable):
            _channels = dict(channels)
            try:
                return {k: LUTModel.model_validate(v) for k, v in _channels.items()}
            except StopIteration:
                pass  # Handle empty iterable
        raise TypeError(
            f"channels must be an int, Mapping, or Iterable, not {type(channels)}"
        )
