from __future__ import annotations

from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Hashable,
    Literal,
    MutableSequence,
    Sequence,
    SupportsIndex,
    cast,
    overload,
)

import cmap
import numpy as np
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QCollapsible, QElidingLabel, QIconifyIcon, ensure_main_thread
from superqt.utils import qthrottled, signals_blocked

from ndv._chunking import Chunker, ChunkResponse, RequestFinished
from ndv.viewer._components import (
    ChannelMode,
    ChannelModeButton,
    DimToggleButton,
    QSpinner,
)

from ._backends import get_canvas
from ._backends._protocols import PImageHandle
from ._data_wrapper import DataWrapper
from ._dims_slider import DimsSliders
from ._lut_control import LutControl

if TYPE_CHECKING:
    from typing import Any, Iterable, TypeAlias

    from qtpy.QtGui import QCloseEvent

    from ._backends._protocols import PCanvas
    from ._dims_slider import DimKey, Indices, Sizes

    ImgKey: TypeAlias = Hashable
    # any mapping of dimensions to sizes
    SizesLike: TypeAlias = Sizes | Iterable[int | tuple[DimKey, int] | Sequence]

MID_GRAY = "#888888"
GRAYS = cmap.Colormap("gray")
DEFAULT_COLORMAPS = [
    cmap.Colormap("green"),
    cmap.Colormap("magenta"),
    cmap.Colormap("cyan"),
    cmap.Colormap("yellow"),
    cmap.Colormap("red"),
    cmap.Colormap("blue"),
    cmap.Colormap("cubehelix"),
    cmap.Colormap("gray"),
]
MONO_CHANNEL = -999999


class Channel(MutableSequence[PImageHandle]):
    def __init__(self, ch_key: int, cmap: cmap.Colormap = GRAYS) -> None:
        self.ch_key = ch_key
        self._handles: list[PImageHandle] = []
        self.cmap = cmap

    @overload
    def __getitem__(self, i: int) -> PImageHandle: ...
    @overload
    def __getitem__(self, i: slice) -> list[PImageHandle]: ...
    def __getitem__(self, i: int | slice) -> PImageHandle | list[PImageHandle]:
        return self._handles[i]

    @overload
    def __setitem__(self, i: SupportsIndex, value: PImageHandle) -> None: ...
    @overload
    def __setitem__(self, i: slice, value: Iterable[PImageHandle]) -> None: ...
    def __setitem__(
        self, i: SupportsIndex | slice, value: PImageHandle | Iterable[PImageHandle]
    ) -> None:
        self._handles[i] = value  # type: ignore

    @overload
    def __delitem__(self, i: int) -> None: ...
    @overload
    def __delitem__(self, i: slice) -> None: ...
    def __delitem__(self, i: int | slice) -> None:
        del self._handles[i]

    def __len__(self) -> int:
        return len(self._handles)

    def insert(self, i: int, value: PImageHandle) -> None:
        self._handles.insert(i, value)


class NDViewer(QWidget):
    """A viewer for ND arrays.

    This widget displays a single slice from an ND array (or a composite of slices in
    different colormaps).  The widget provides sliders to select the slice to display,
    and buttons to control the display mode of the channels.

    An important concept in this widget is the "index".  The index is a mapping of
    dimensions to integers or slices that define the slice of the data to display.  For
    example, a numpy slice of `[0, 1, 5:10]` would be represented as
    `{0: 0, 1: 1, 2: slice(5, 10)}`, but dimensions can also be named, e.g.
    `{'t': 0, 'c': 1, 'z': slice(5, 10)}`. The index is used to select the data from
    the datastore, and to determine the position of the sliders.

    The flow of data is as follows:

    - The user sets the data using the `set_data` method. This will set the number
      and range of the sliders to the shape of the data, and display the first slice.
    - The user can then use the sliders to select the slice to display. The current
      slice is defined as a `Mapping` of `{dim -> int|slice}` and can be retrieved
      with the `_dims_sliders.value()` method.  To programmatically set the current
      position, use the `setIndex` method. This will set the values of the sliders,
      which in turn will trigger the display of the new slice via the
      `_request_data_for_index` method.
    - `_request_data_for_index` is an asynchronous method that retrieves the data for
      the given index from the datastore (using `_isel`) and queues the
      `_draw_chunk` method to be called when the data is ready. The logic
      for extracting data from the datastore is defined in `_data_wrapper.py`, which
      handles idiosyncrasies of different datastores (e.g. xarray, tensorstore, etc).
    - `_draw_chunk` is called when the data is ready, and updates the image.
      Note that if the slice is multidimensional, the data will be reduced to 2D using
      max intensity projection (and double-clicking on any given dimension slider will
      turn it into a range slider allowing a projection to be made over that dimension).
    - The image is displayed on the canvas, which is an object that implements the
      `PCanvas` protocol (mostly, it has an `add_image` method that returns a handle
      to the added image that can be used to update the data and display). This
      small abstraction allows for various backends to be used (e.g. vispy, pygfx, etc).

    Parameters
    ----------
    data : Any
        The data to display.  This can be any duck-like ND array, including numpy, dask,
        xarray, jax, tensorstore, zarr, etc.  You can add support for new datastores by
        subclassing `DataWrapper` and implementing the required methods.  See
        `DataWrapper` for more information.
    parent : QWidget, optional
        The parent widget of this widget.
    channel_axis : Hashable, optional
        The axis that represents the channels in the data.  If not provided, this will
        be guessed from the data.
    channel_mode : ChannelMode, optional
        The initial mode for displaying the channels. If not provided, this will be
        set to ChannelMode.MONO.
    """

    def __init__(
        self,
        data: DataWrapper | Any,
        *,
        colormaps: Iterable[cmap._colormap.ColorStopsLike] | None = None,
        parent: QWidget | None = None,
        channel_axis: int | None = None,
        channel_mode: ChannelMode | str = ChannelMode.MONO,
    ):
        super().__init__(parent=parent)

        # ATTRIBUTES ----------------------------------------------------

        # mapping of key to a list of objects that control image nodes in the canvas
        self._channels: dict[int, Channel] = {}

        # mapping of same keys to the LutControl objects control image display props
        self._lut_ctrls: dict[int, LutControl] = {}

        # the set of dimensions we are currently visualizing (e.g. (-2, -1) for 2D)
        # this is used to control which dimensions have sliders and the behavior
        # of isel when selecting data from the datastore
        self._visualized_dims: set[DimKey] = set()

        # the axis that represents the channels in the data
        self._channel_axis = channel_axis
        self._channel_mode: ChannelMode = None  # type: ignore # set in set_channel_mode
        # colormaps that will be cycled through when displaying composite images
        if colormaps is not None:
            self._cmaps = [cmap.Colormap(c) for c in colormaps]
        else:
            self._cmaps = DEFAULT_COLORMAPS
        self._cmap_cycle = cycle(self._cmaps)

        # number of dimensions to display
        self._ndims: Literal[2, 3] = 2
        self._chunker = Chunker(
            None,
            # IMPORTANT
            # chunking here will determine how non-visualized dims are reduced
            # so chunkshape will need to change based on the set of visualized dims
            chunks=(20, 100, 32, 32),
            on_ready=self._draw_chunk,
        )

        # WIDGETS ----------------------------------------------------

        # the button that controls the display mode of the channels
        self._channel_mode_btn = ChannelModeButton(self)
        self._channel_mode_btn.clicked.connect(self.set_channel_mode)
        # button to reset the zoom of the canvas
        self._set_range_btn = QPushButton(
            QIconifyIcon("fluent:full-screen-maximize-24-filled"), "", self
        )
        self._set_range_btn.clicked.connect(self._on_set_range_clicked)

        # button to change number of displayed dimensions
        self._ndims_btn = DimToggleButton(self)
        self._ndims_btn.clicked.connect(self._toggle_3d)

        # place to display dataset summary
        self._data_info_label = QElidingLabel("", parent=self)
        self._progress_spinner = QSpinner(self)

        # place to display arbitrary text
        self._hover_info_label = QLabel("", self)
        # the canvas that displays the images
        self._canvas: PCanvas = get_canvas()(self._hover_info_label.setText)
        self._canvas.set_ndim(self._ndims)

        # the sliders that control the index of the displayed image
        self._dims_sliders = DimsSliders(self)
        self._dims_sliders.valueChanged.connect(
            qthrottled(self._request_data_for_index, 20, leading=True)
        )

        self._lut_drop = QCollapsible("LUTs", self)
        self._lut_drop.setCollapsedIcon(QIconifyIcon("bi:chevron-down", color=MID_GRAY))
        self._lut_drop.setExpandedIcon(QIconifyIcon("bi:chevron-up", color=MID_GRAY))
        lut_layout = cast("QVBoxLayout", self._lut_drop.layout())
        lut_layout.setContentsMargins(0, 1, 0, 1)
        lut_layout.setSpacing(0)
        if (
            hasattr(self._lut_drop, "_content")
            and (layout := self._lut_drop._content.layout()) is not None
        ):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

        # LAYOUT -----------------------------------------------------

        self._btns = btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(0)
        btns.addStretch()
        btns.addWidget(self._channel_mode_btn)
        btns.addWidget(self._ndims_btn)
        btns.addWidget(self._set_range_btn)

        info = QHBoxLayout()
        info.setContentsMargins(0, 0, 0, 2)
        info.setSpacing(0)
        info.addWidget(self._data_info_label)
        info.addWidget(self._progress_spinner)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(info)
        layout.addWidget(self._canvas.qwidget(), 1)
        layout.addWidget(self._hover_info_label)
        layout.addWidget(self._dims_sliders)
        layout.addWidget(self._lut_drop)
        layout.addLayout(btns)

        # SETUP ------------------------------------------------------

        self.set_channel_mode(channel_mode)
        if data is not None:
            self.set_data(data)

    # ------------------- PUBLIC API ----------------------------
    @property
    def dims_sliders(self) -> DimsSliders:
        """Return the DimsSliders widget."""
        return self._dims_sliders

    @property
    def data_wrapper(self) -> DataWrapper:
        """Return the DataWrapper object around the datastore."""
        return self._data_wrapper

    @property
    def data(self) -> Any:
        """Return the data backing the view."""
        return self._data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data backing the view."""
        raise AttributeError("Cannot set data directly. Use `set_data` method.")

    def set_data(
        self, data: DataWrapper | Any, *, initial_index: Indices | None = None
    ) -> None:
        """Set the datastore, and, optionally, the sizes of the data.

        Properties
        ----------
        data : DataWrapper | Any
            The data to display.  This can be any duck-like ND array, including numpy,
            dask, xarray, jax, tensorstore, zarr, etc.  You can add support for new
            datastores by subclassing `DataWrapper` and implementing the required
            methods. If a `DataWrapper` instance is passed, it is used directly.
            See `DataWrapper` for more information.
        initial_index : Indices | None
            The initial index to display.  This is a mapping of dimensions to integers
            or slices that define the slice of the data to display.  If not provided,
            the initial index will be set to the middle of the data.
        """
        # store the data
        self._clear_images()

        self._data_wrapper = DataWrapper.create(data)
        self._chunker.data_wrapper = self._data_wrapper
        if chunks := self._data_wrapper.chunks():
            # temp hack ... always group non-visible channels
            chunks = list(chunks)
            chunks[:-2] = (1000,) * len(chunks[:-2])
            self._chunker.chunks = tuple(chunks)

        # set channel axis
        self._channel_axis = self._data_wrapper.guess_channel_axis()
        self._chunker.channel_axis = self._channel_axis

        # update the dimensions we are visualizing
        sizes = self._data_wrapper.sizes()
        visualized_dims = list(sizes)[-self._ndims :]
        self.set_visualized_dims(visualized_dims)

        # update the range of all the sliders to match the sizes we set above
        with signals_blocked(self._dims_sliders):
            self._update_slider_ranges()

        # redraw
        if initial_index is None:
            idx = {k: int(v // 2) for k, v in sizes.items() if k not in visualized_dims}
        else:
            if not isinstance(initial_index, dict):  # pragma: no cover
                raise TypeError("initial_index must be a dict")
            idx = initial_index
        self.set_current_index(idx)

        # update the data info label
        self._data_info_label.setText(self._data_wrapper.summary_info())

    def set_visualized_dims(self, dims: Iterable[DimKey]) -> None:
        """Set the dimensions that will be visualized.

        This dims will NOT have sliders associated with them.
        """
        self._visualized_dims = set(dims)
        for d in self._dims_sliders._sliders:
            self._dims_sliders.set_dimension_visible(d, d not in self._visualized_dims)
        for d in self._visualized_dims:
            self._dims_sliders.set_dimension_visible(d, False)

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        """Set the number of dimensions to display."""
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")

        self._ndims = ndim
        self._canvas.set_ndim(ndim)

        # set the visibility of the last non-channel dimension
        sizes = list(self._data_wrapper.sizes())
        if self._channel_axis is not None:
            sizes = [x for x in sizes if x != self._channel_axis]
        if len(sizes) >= 3:
            dim3 = sizes[-3]
            self._dims_sliders.set_dimension_visible(dim3, True if ndim == 2 else False)

        # clear image handles and redraw
        if self._channels:
            self._clear_images()
            self._request_data_for_index(self._dims_sliders.value())

    def set_channel_mode(self, mode: ChannelMode | str | None = None) -> None:
        """Set the mode for displaying the channels.

        In "composite" mode, the channels are displayed as a composite image, using
        self._channel_axis as the channel axis. In "grayscale" mode, each channel is
        displayed separately. (If mode is None, the current value of the
        channel_mode_picker button is used)

        Parameters
        ----------
        mode : ChannelMode | str | None
            The mode to set, must be one of 'composite' or 'mono'.
        """
        # bool may happen when called from the button clicked signal
        if mode is None or isinstance(mode, bool):
            mode = self._channel_mode_btn.mode()
        else:
            mode = ChannelMode(mode)
            self._channel_mode_btn.setMode(mode)
        if mode == self._channel_mode:
            return

        self._channel_mode = mode
        self._cmap_cycle = cycle(self._cmaps)  # reset the colormap cycle
        if self._channel_axis is not None:
            # set the visibility of the channel slider
            self._dims_sliders.set_dimension_visible(
                self._channel_axis, mode != ChannelMode.COMPOSITE
            )

        if self._channels:
            self._clear_images()
            self._request_data_for_index(self._dims_sliders.value())

    def set_current_index(self, index: Indices | None = None) -> None:
        """Set the index of the displayed image.

        `index` is a mapping of dimensions to integers or slices that define the slice
        of the data to display.  For example, a numpy slice of `[0, 1, 5:10]` would be
        represented as `{0: 0, 1: 1, 2: slice(5, 10)}`, but dimensions can also be
        named, e.g. `{'t': 0, 'c': 1, 'z': slice(5, 10)}` if the data has named
        dimensions.

        Note, calling `.set_current_index()` with no arguments will force the widget
        to redraw the current slice.
        """
        self._dims_sliders.setValue(index or {})

    # camelCase aliases

    dimsSliders = dims_sliders
    setChannelMode = set_channel_mode
    setData = set_data
    setCurrentIndex = set_current_index
    setVisualizedDims = set_visualized_dims

    # ------------------- PRIVATE METHODS ----------------------------

    def _toggle_3d(self) -> None:
        self.set_ndim(3 if self._ndims == 2 else 2)

    def _update_slider_ranges(self) -> None:
        """Set the maximum values of the sliders.

        If `sizes` is not provided, sizes will be inferred from the datastore.
        """
        maxes = self._data_wrapper.sizes()
        self._dims_sliders.setMaxima({k: v - 1 for k, v in maxes.items()})

        # FIXME: this needs to be moved and made user-controlled
        for dim in list(maxes.keys())[-self._ndims :]:
            self._dims_sliders.set_dimension_visible(dim, False)

    def _on_set_range_clicked(self) -> None:
        # using method to swallow the parameter passed by _set_range_btn.clicked
        self._canvas.set_range()

    def _image_key(self, index: Indices) -> ImgKey:
        """Return the key for image handle(s) corresponding to `index`."""
        if self._channel_mode == ChannelMode.COMPOSITE:
            val = index.get(self._channel_axis, 0)
            if isinstance(val, slice):
                return (val.start, val.stop)
            return val
        return 0

    def _request_data_for_index(self, index: Indices) -> None:
        """Retrieve data for `index` from datastore and update canvas image(s).

        This is the first step in updating the displayed image, it is triggered by
        the valueChanged signal from the sliders.

        This will pull the data from the datastore using the given index, and update
        the image handle(s) with the new data.  This method is *asynchronous*.  It
        makes a request for the new data slice and queues _on_data_future_done to be
        called when the data is ready.
        """
        print(f"\n--------\nrequesting index {index}", self._channel_axis)
        if (
            self._channel_mode == ChannelMode.COMPOSITE
            and self._channel_axis is not None
        ):
            index = {**index, self._channel_axis: slice(None)}
        self._progress_spinner.show()
        # TODO: don't request channels not being displayed
        # TODO: don't request if the data is already in the cache
        self._chunker.request_index(index, ndims=self._ndims)

    @ensure_main_thread  # type: ignore
    def _draw_chunk(self, chunk: ChunkResponse) -> None:
        """Actually update the image handle(s) with the (sliced) data.

        By this point, data should be sliced from the underlying datastore.  Any
        dimensions remaining that are more than the number of visualized dimensions
        (currently just 2D) will be reduced using max intensity projection (currently).
        """
        if chunk is RequestFinished:  # fix typing
            self._progress_spinner.hide()
            for lut in self._lut_ctrls.values():
                lut.update_autoscale()
            return

        if self._channel_mode == ChannelMode.MONO:
            ch_key = MONO_CHANNEL
        else:
            ch_key = chunk.channel_index

        data = chunk.data
        if data.ndim == 2:
            return
        # TODO: Channel object creation could be moved.
        # having it here is the laziest... but means that the order of arrival
        # of the chunks will determine the order of the channels in the LUTS
        # (without additional logic to sort them by index, etc.)
        if (handles := self._channels.get(ch_key)) is None:
            handles = self._create_channel(ch_key)

        if not handles:
            if data.ndim == 2:
                handles.append(self._canvas.add_image(data, cmap=handles.cmap))
            elif data.ndim == 3:
                empty = np.empty((60, 256, 256), dtype=np.uint16)
                handles.append(self._canvas.add_volume(empty, cmap=handles.cmap))

        handles[0].set_data(data, chunk.offset)
        self._canvas.refresh()

    def _create_channel(self, ch_key: int) -> Channel:
        # improve this
        cmap = GRAYS if ch_key == MONO_CHANNEL else next(self._cmap_cycle)

        self._channels[ch_key] = channel = Channel(ch_key, cmap=cmap)
        self._lut_ctrls[ch_key] = lut = LutControl(
            channel,
            f"Ch {ch_key}",
            self,
            cmaplist=self._cmaps + DEFAULT_COLORMAPS,
            cmap=cmap,
        )
        self._lut_drop.addWidget(lut)
        return channel

    def _clear_images(self) -> None:
        """Remove all images from the canvas."""
        for handles in self._channels.values():
            for handle in handles:
                handle.remove()
        self._channels.clear()

        # clear the current LutControls as well
        for c in self._lut_ctrls.values():
            cast("QVBoxLayout", self.layout()).removeWidget(c)
            c.deleteLater()
        self._lut_ctrls.clear()

    def _is_idle(self) -> bool:
        """Return True if no futures are running. Used for testing, and debugging."""
        return bool(self._chunker.pending_futures)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self._chunker.shutdown()
        super().closeEvent(a0)
