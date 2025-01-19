from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING, Literal, cast

import cmap
import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QCollapsible, QElidingLabel, QIconifyIcon, ensure_main_thread
from superqt.utils import qthrottled, signals_blocked

from ndv.views import get_array_canvas_class

from ._old_data_wrapper import DataWrapper
from ._qt._components import (
    ChannelMode,
    ChannelModeButton,
    DimToggleButton,
    QSpinner,
    ROIButton,
)
from ._qt._dims_slider import DimsSliders
from ._qt._lut_control import LutControl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from concurrent.futures import Future
    from typing import Any, Callable, TypeAlias

    from qtpy.QtCore import QObject
    from qtpy.QtGui import QCloseEvent, QKeyEvent

    from ndv.views.bases._graphics._canvas import ArrayCanvas
    from ndv.views.bases._graphics._canvas_elements import (
        CanvasElement,
        ImageHandle,
        RoiHandle,
    )

    DimKey = int
    Index = int | slice
    Indices = Mapping[Any, Index]
    Sizes = Mapping[Any, int]

    ImgKey: TypeAlias = Any
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
ALL_CHANNELS = slice(None)


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
      `_update_data_for_index` method.
    - `_update_data_for_index` is an asynchronous method that retrieves the data for
      the given index from the datastore (using `_isel`) and queues the
      `_on_data_slice_ready` method to be called when the data is ready. The logic
      for extracting data from the datastore is defined in `_data_wrapper.py`, which
      handles idiosyncrasies of different datastores (e.g. xarray, tensorstore, etc).
    - `_on_data_slice_ready` is called when the data is ready, and updates the image.
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
        data: DataWrapper | Any | None = None,
        *,
        colormaps: Iterable[cmap._colormap.ColorStopsLike] | None = None,
        parent: QWidget | None = None,
        channel_axis: DimKey | None = None,
        channel_mode: ChannelMode | str = ChannelMode.MONO,
    ):
        super().__init__(parent=parent)

        # ATTRIBUTES ----------------------------------------------------
        self._data_wrapper: DataWrapper | None = None

        # mapping of key to a list of objects that control image nodes in the canvas
        self._img_handles: defaultdict[ImgKey, list[ImageHandle]] = defaultdict(list)
        # mapping of same keys to the LutControl objects control image display props
        self._lut_ctrls: dict[ImgKey, LutControl] = {}
        self._lut_ctrl_state: dict[ImgKey, dict] = {}
        # the set of dimensions we are currently visualizing (e.g. XY)
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
        # the last future that was created by _update_data_for_index
        self._last_future: Future | None = None

        # number of dimensions to display
        self._ndims: Literal[2, 3] = 2

        # Canvas selection
        self._selection: CanvasElement | None = None
        # ROI
        self._roi: RoiHandle | None = None

        # WIDGETS ----------------------------------------------------

        # the button that controls the display mode of the channels
        self._channel_mode_btn = ChannelModeButton(self)
        self._channel_mode_btn.clicked.connect(self.set_channel_mode)
        # button to reset the zoom of the canvas
        self._set_range_btn = QPushButton(
            QIconifyIcon("fluent:full-screen-maximize-24-filled"), "", self
        )
        self._set_range_btn.clicked.connect(self._on_set_range_clicked)
        # button to draw ROIs
        self._add_roi_btn = ROIButton()
        self._add_roi_btn.toggled.connect(self._on_add_roi_clicked)

        # button to change number of displayed dimensions
        self._ndims_btn = DimToggleButton(self)
        self._ndims_btn.clicked.connect(self._toggle_3d)

        # place to display dataset summary
        self._data_info_label = QElidingLabel("", parent=self)
        self._progress_spinner = QSpinner(self)

        # place to display arbitrary text
        self._hover_info_label = QLabel("", self)
        # the canvas that displays the images
        self._canvas: ArrayCanvas = get_array_canvas_class()()
        self._canvas.set_ndim(self._ndims)
        self._qcanvas = self._canvas.frontend_widget()

        # Install an event filter so we can intercept mouse/key events
        self._qcanvas.installEventFilter(self)

        # the sliders that control the index of the displayed image
        self._dims_sliders = DimsSliders(self)
        self._dims_sliders.valueChanged.connect(
            qthrottled(self._update_data_for_index, 20, leading=True)
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
        btns.addWidget(self._add_roi_btn)

        info_widget = QWidget()
        info = QHBoxLayout(info_widget)
        info.setContentsMargins(0, 0, 0, 2)
        info.setSpacing(0)
        info.addWidget(self._data_info_label)
        info.addWidget(self._progress_spinner)
        info_widget.setFixedHeight(16)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(info_widget)
        layout.addWidget(self._qcanvas, 1)
        layout.addWidget(self._hover_info_label)
        layout.addWidget(self._dims_sliders)
        layout.addWidget(self._lut_drop)
        layout.addLayout(btns)

        # SETUP ------------------------------------------------------

        self.set_channel_mode(channel_mode)
        self.set_data(data)

    # ------------------- PUBLIC API ----------------------------
    @property
    def dims_sliders(self) -> DimsSliders:
        """Return the DimsSliders widget."""
        return self._dims_sliders

    @property
    def data_wrapper(self) -> DataWrapper | None:
        """Return the DataWrapper object around the datastore."""
        return self._data_wrapper

    @property
    def data(self) -> Any:
        """Return the data backing the view."""
        if self._data_wrapper is None:
            return None
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
        # clear current data
        if data is None:
            self._data_wrapper = None
            self._clear_images()
            self._dims_sliders.clear()
            self._data_info_label.setText("")
            return

        # store the data
        self._data_wrapper = DataWrapper.create(data)

        # set channel axis
        self._channel_axis = self._data_wrapper.guess_channel_axis()

        # update the dimensions we are visualizing
        sizes = self._data_wrapper.sizes()
        visualized_dims = list(sizes)[-self._ndims :]
        self.set_visualized_dims(visualized_dims)

        # update the range of all the sliders to match the sizes we set above
        with signals_blocked(self._dims_sliders):
            self._update_slider_ranges()

        # redraw
        if initial_index is None:
            idx = self._dims_sliders.value() or {
                k: int(v // 2) for k, v in sizes.items()
            }
        else:
            if not isinstance(initial_index, dict):  # pragma: no cover
                raise TypeError("initial_index must be a dict")
            idx = initial_index
        with signals_blocked(self._dims_sliders):
            self.set_current_index(idx)
        # update the data info label
        self._data_info_label.setText(self._data_wrapper.summary_info())
        self.refresh()

    def set_roi(
        self,
        vertices: list[tuple[float, float]] | None = None,
        color: Any = None,
        border_color: Any = None,
    ) -> None:
        """Set the properties of the ROI overlaid on the displayed data.

        Properties
        ----------
        vertices : list[tuple[float, float]] | None
            The vertices of the ROI.
        color : str, tuple, list, array, Color, or int
            The fill color.  Can be any "ColorLike".
        border_color : str, tuple, list, array, Color, or int
            The border color.  Can be any "ColorLike".
        """
        # Remove the old ROI
        if self._roi:
            self._roi.remove()
        color = cmap.Color(color) if color is not None else None
        border_color = cmap.Color(border_color) if border_color is not None else None
        self._roi = self._canvas.add_roi(
            vertices=vertices, color=color, border_color=border_color
        )

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

        if self._data_wrapper is None:
            return

        # set the visibility of the last non-channel dimension
        sizes = list(self._data_wrapper.sizes())
        if self._channel_axis is not None:
            sizes = [x for x in sizes if x != self._channel_axis]
        if len(sizes) >= 3:
            dim3 = sizes[-3]
            self._dims_sliders.set_dimension_visible(dim3, True if ndim == 2 else False)

        # clear image handles and redraw
        self.refresh()

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

        self.refresh()

    def refresh(self) -> None:
        """Refresh the canvas."""
        self._clear_images()
        self._update_data_for_index(self._dims_sliders.value())

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

        # Disable ROIs in 3D (for now)
        self._add_roi_btn.setEnabled(self._ndims == 2)
        # FIXME: When toggling 2D again, ROIs cannot be selected
        if self._roi:
            self._roi.set_visible(self._ndims == 2)

    def _update_slider_ranges(self) -> None:
        """Set the maximum values of the sliders.

        If `sizes` is not provided, sizes will be inferred from the datastore.
        """
        if self._data_wrapper is None:
            return

        maxes = self._data_wrapper.sizes()
        self._dims_sliders.setMaxima({k: v - 1 for k, v in maxes.items()})

        # FIXME: this needs to be moved and made user-controlled
        for dim in list(maxes.keys())[-self._ndims :]:
            self._dims_sliders.set_dimension_visible(dim, False)

    def _on_set_range_clicked(self) -> None:
        # using method to swallow the parameter passed by _set_range_btn.clicked
        self._canvas.set_range()

    def _on_add_roi_clicked(self, checked: bool) -> None:
        if checked:
            # Add new roi
            self.set_roi()

    def _image_key(self, index: Indices) -> ImgKey:
        """Return the key for image handle(s) corresponding to `index`."""
        if self._channel_mode == ChannelMode.COMPOSITE:
            val = index.get(self._channel_axis, 0)
            if isinstance(val, slice):
                return (val.start, val.stop)
            return val
        return 0

    def _update_data_for_index(self, index: Indices) -> None:
        """Retrieve data for `index` from datastore and update canvas image(s).

        This will pull the data from the datastore using the given index, and update
        the image handle(s) with the new data.  This method is *asynchronous*.  It
        makes a request for the new data slice and queues _on_data_future_done to be
        called when the data is ready.
        """
        if self._data_wrapper is None:
            return
        if (
            self._channel_axis is not None
            and self._channel_mode == ChannelMode.COMPOSITE
            and self._channel_axis in (sizes := self._data_wrapper.sizes())
        ):
            indices: list[Indices] = [
                {**index, self._channel_axis: i}
                for i in range(sizes[self._channel_axis])
            ]
        else:
            indices = [index]

        if self._last_future:
            self._last_future.cancel()

        # don't request any dimensions that are not visualized
        indices = [
            {k: v for k, v in idx.items() if k not in self._visualized_dims}
            for idx in indices
        ]
        try:
            self._last_future = f = self._data_wrapper.isel_async(indices)
        except Exception as e:
            raise type(e)(f"Failed to index data with {index}: {e}") from e

        self._progress_spinner.show()
        f.add_done_callback(self._on_data_slice_ready)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if self._last_future is not None:
            self._last_future.cancel()
            self._last_future = None
        super().closeEvent(a0)

    @ensure_main_thread  # type: ignore
    def _on_data_slice_ready(
        self, future: Future[Iterable[tuple[Indices, np.ndarray]]]
    ) -> None:
        """Update the displayed image for the given index.

        Connected to the future returned by _isel.
        """
        # NOTE: removing the reference to the last future here is important
        # because the future has a reference to this widget in its _done_callbacks
        # which will prevent the widget from being garbage collected if the future
        self._last_future = None
        self._progress_spinner.hide()
        if future.cancelled():
            return

        for idx, datum in future.result():
            self._update_canvas_data(datum, idx)
        self._canvas.refresh()

    def _update_canvas_data(self, data: np.ndarray, index: Indices) -> None:
        """Actually update the image handle(s) with the (sliced) data.

        By this point, data should be sliced from the underlying datastore.  Any
        dimensions remaining that are more than the number of visualized dimensions
        (currently just 2D) will be reduced using max intensity projection (currently).
        """
        imkey = self._image_key(index)
        datum = self._reduce_data_for_display(data)
        if handles := self._img_handles[imkey]:
            for handle in handles:
                handle.set_data(datum)
            if ctrl := self._lut_ctrls.get(imkey, None):
                ctrl.update_autoscale()
        else:
            cm = (
                next(self._cmap_cycle)
                if self._channel_mode == ChannelMode.COMPOSITE
                else GRAYS
            )
            if datum.ndim == 2:
                handle = self._canvas.add_image(datum)
                handle.set_cmap(cm)
                handles.append(handle)
            elif datum.ndim == 3:
                handle = self._canvas.add_volume(datum)
                handle.set_cmap(cm)
                handles.append(handle)
            if imkey not in self._lut_ctrls:
                ch_index = index.get(self._channel_axis, 0)
                self._lut_ctrls[imkey] = c = LutControl(
                    f"Ch {ch_index}",
                    handles,
                    self,
                    cmaplist=self._cmaps + DEFAULT_COLORMAPS,
                )
                # HACK:
                # this is a temporary "better than nothing" to persist LUT control
                # state for a given channel across instances of LutControl widget.
                # we need a better model, detached from the view/widget to manage this.
                if imkey in self._lut_ctrl_state:
                    c._set_state(self._lut_ctrl_state[imkey])
                self._lut_drop.addWidget(c)

    def _reduce_data_for_display(
        self, data: np.ndarray, reductor: Callable[..., np.ndarray] = np.max
    ) -> np.ndarray:
        """Reduce the number of dimensions in the data for display.

        This function takes a data array and reduces the number of dimensions to
        the max allowed for display. The default behavior is to reduce the smallest
        dimensions, using np.max.  This can be improved in the future.

        This also coerces 64-bit data to 32-bit data.
        """
        # TODO
        # - allow dimensions to control how they are reduced (as opposed to just max)
        # - for better way to determine which dims need to be reduced (currently just
        #   the smallest dims)
        data = data.squeeze()
        visualized_dims = self._ndims
        if extra_dims := data.ndim - visualized_dims:
            shapes = sorted(enumerate(data.shape), key=lambda x: x[1])
            smallest_dims = tuple(i for i, _ in shapes[:extra_dims])
            data = reductor(data, axis=smallest_dims)

        if data.dtype.itemsize > 4:  # More than 32 bits
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.int32)
            else:
                data = data.astype(np.float32)
        return data

    def _clear_images(self) -> None:
        """Remove all images from the canvas."""
        for handles in self._img_handles.values():
            for handle in handles:
                handle.remove()
        self._img_handles.clear()

        # clear the current LutControls as well
        for k, c in self._lut_ctrls.items():
            # HACK:
            # this is a temporary "better than nothing" to persist LUT control
            # state for a given channel across instances of LutControl widget.
            # we need a better model, detached from the view/widget to manage this.
            self._lut_ctrl_state[k] = c._get_state()
            cast("QVBoxLayout", self.layout()).removeWidget(c)
            c.deleteLater()
        self._lut_ctrls.clear()

    def _is_idle(self) -> bool:
        """Return True if no futures are running. Used for testing, and debugging."""
        return self._last_future is None

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Event filter installed on the canvas to handle mouse events."""
        if event is None:
            return False  # pragma: no cover

        # here is where we get a chance to intercept mouse events before passing them
        # to the canvas. Return `True` to prevent the event from being passed to
        # the backend widget.
        intercept = False
        # use children in case backend has a subwidget stealing events.
        if obj is self._qcanvas or obj in (self._qcanvas.children()):
            if isinstance(event, QMouseEvent):
                intercept |= self._canvas_mouse_event(event)
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(cast("QKeyEvent", event))
        return intercept

    def _canvas_mouse_event(self, ev: QMouseEvent) -> bool:
        intercept = False
        if ev.type() == QEvent.Type.MouseButtonPress:
            if self._add_roi_btn.isChecked():
                intercept |= self._begin_roi(ev)
            intercept |= self._press_element(ev)
            return intercept
        if ev.type() == QEvent.Type.MouseMove:
            intercept = self._move_selection(ev)
            intercept |= self._update_hover_info(ev)
            intercept |= self._update_cursor(ev)
            return intercept
        if ev.type() == QEvent.Type.MouseButtonRelease:
            return self._update_roi_button(ev)
        return False

    # FIXME: This is ugly
    def _begin_roi(self, event: QMouseEvent) -> bool:
        if self._roi:
            ev_pos = event.position()
            pos = self._canvas.canvas_to_world((ev_pos.x(), ev_pos.y()))
            self._roi.move(pos)
            self._roi.set_visible(True)
        return False

    def _press_element(self, event: QMouseEvent) -> bool:
        ev_pos = (event.position().x(), event.position().y())
        pos = self._canvas.canvas_to_world(ev_pos)
        # TODO why does the canvas need this point untransformed??
        elements = self._canvas.elements_at(ev_pos)
        # Deselect prior selection before editing new selection
        if self._selection:
            self._selection.set_selected(False)
        for e in elements:
            if e.can_select():
                e.start_move(pos)
                # Select new selection
                self._selection = e
                self._selection.set_selected(True)
                return False
        return False

    def _move_selection(self, event: QMouseEvent) -> bool:
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self._selection and self._selection.selected():
                ev_pos = event.pos()
                pos = self._canvas.canvas_to_world((ev_pos.x(), ev_pos.y()))
                self._selection.move(pos)
                # If we are moving the object, we don't want to move the camera
                return True
        return False

    def _update_cursor(self, event: QMouseEvent) -> bool:
        # Avoid changing the cursor when dragging
        if event.buttons() != Qt.MouseButton.NoButton:
            return False
        # When "creating" a ROI, use CrossCursor
        if self._add_roi_btn.isChecked():
            self._qcanvas.setCursor(Qt.CursorShape.CrossCursor)
            return False
        # If any local elements have a preference, use it
        pos = (event.pos().x(), event.pos().y())
        for e in self._canvas.elements_at(pos):
            if (pref := e.cursor_at(pos)) is not None:
                self._qcanvas.setCursor(pref.to_qt())
                return False
        # Otherwise, normal cursor
        self._qcanvas.setCursor(Qt.CursorShape.ArrowCursor)
        return False

    def _update_hover_info(self, event: QMouseEvent) -> bool:
        """Update text of hover_info_label with data value(s) at point."""
        point = event.pos()
        x, y, _z = self._canvas.canvas_to_world((point.x(), point.y()))
        # TODO: handle 3D data
        if (x < 0 or y < 0) or self._ndims == 3:  # pragma: no cover
            self._hover_info_label.setText("")
            return False

        x = int(x)
        y = int(y)
        text = f"[{y}, {x}]"
        # TODO: Can we use self._canvas.elements_at?
        for n, handles in enumerate(self._img_handles.values()):
            channels = []
            for handle in handles:
                try:
                    # here, we're retrieving the value from the in-memory data
                    # stored by the backend visual, rather than querying the data itself
                    # this is a quick workaround to get the value without having to
                    # worry about higher dimensions in the data source (since the
                    # texture has already been reduced to 2D). But a more complete
                    # implementation would gather the full current nD index and query
                    # the data source directly.
                    value = handle.data()[y, x]
                    if isinstance(value, (np.floating, float)):
                        value = f"{value:.2f}"
                    channels.append(f" {n}: {value}")
                except IndexError:
                    # we're out of bounds
                    # if we eventually have multiple image sources with different
                    # extents, this will need to be handled.  here, we just skip
                    self._hover_info_label.setText("")
                    return False
                break  # only getting one handle per channel
            text += ",".join(channels)
        self._hover_info_label.setText(text)
        return False

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is None:
            return
        if a0.key() == Qt.Key.Key_Delete and self._selection is not None:
            self._selection.remove()
            self._selection = None

    def _update_roi_button(self, event: QMouseEvent) -> bool:
        if self._add_roi_btn.isChecked():
            self._add_roi_btn.click()
        return False
