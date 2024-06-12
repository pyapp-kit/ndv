from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import qthrottled

from ndv._chunk_executor import Chunker, ChunkFuture
from ndv.viewer2._backends import get_canvas
from ndv.viewer2._dims_slider import DimsSliders
from ndv.viewer2._state import ViewerState

if TYPE_CHECKING:
    from ndv.viewer2._backends.protocols import PCanvas, PImageHandle


class NDViewer(QWidget):
    def __init__(self, data: Any, *, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self._state = ViewerState(visualized_indices=(0, 2, 3))
        self._chunker = Chunker()
        self._channels: dict[int | None, PImageHandle] = {}

        self._canvas: PCanvas = get_canvas()(lambda x: None)
        self._canvas.set_ndim(self._state.ndim)

        # the sliders that control the index of the displayed image
        self._dims_sliders = DimsSliders(self)
        self._dims_sliders.valueChanged.connect(
            qthrottled(self._request_data_for_index, 20, leading=True)
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._canvas.qwidget(), 1)
        layout.addWidget(self._dims_sliders, 0)

        if data is not None:
            self.set_data(data)

    def __del__(self) -> None:
        self._chunker.shutdown(cancel_futures=True, wait=False)

    def set_data(self, data: Any) -> None:
        self._data = data
        self._dims_sliders.setMaxima(
            {i: data.shape[i] - 1 for i in range(len(data.shape))}
        )

    def set_current_index(self, index: Mapping[int | str, int | slice]) -> None:
        """Set the currentl displayed index."""
        self._state.current_index = index
        self.refresh()

    def refresh(self) -> None:
        self._request_data_for_index(self._state.current_index)

    def _norm_index(self, index: int | str) -> int:
        """Remove string keys from index."""
        # TODO: this is a temporary solution
        # the datawrapper __getitem__ should handle this
        dim_names = ()
        if index in dim_names:
            return dim_names.index(index)
        elif isinstance(index, int):
            return index
        raise ValueError(f"Invalid index: {index}")

    def _request_data_for_index(self, index: Mapping[int | str, int | slice]) -> None:
        ndim = len(self._data.shape)

        # determine chunk shape
        # only visualized dimensions are chunked
        chunk_size = 128  # TODO: pick bettter
        chunk_shape: list[int | None] = [None] * ndim
        visualized = [self._norm_index(dim) for dim in self._state.visualized_indices]
        for dim in range(ndim):
            if dim in visualized:
                chunk_shape[dim] = chunk_size

        index = {self._norm_index(k): v for k, v in index.items()}
        for v in visualized:
            if isinstance(index.get(v), int):
                del index[v]

        if not index:
            return

        # clear existing handles
        for handle in self._channels.values():
            handle.clear()
        for future in self._chunker.request_chunks(
            data=self._data,
            index=index,
            chunk_shape=chunk_shape,
            cancel_existing=True,
        ):
            future.add_done_callback(self._draw_chunk)

    @ensure_main_thread  # type: ignore
    def _draw_chunk(self, future: ChunkFuture) -> None:
        if future.cancelled():
            return
        if future.exception():
            print("ERROR: ", future.exception())
            return

        chunk = future.result()
        data = chunk.data
        offset = chunk.offset

        if self._state.channel_index is None:
            channel_index = None
        else:
            channel_index = offset[self._norm_index(self._state.channel_index)]

        visualized = [self._norm_index(dim) for dim in self._state.visualized_indices]
        offset = tuple(offset[i] for i in visualized)

        if data.ndim == 2:
            return

        if not (handle := self._channels.get(channel_index)):
            full_shape = self._data.shape
            texture_shape = tuple(
                full_shape[self._norm_index(i)] for i in self._state.visualized_indices
            )
            empty = np.empty(texture_shape, dtype=chunk.data.dtype)
            self._channels[channel_index] = handle = self._canvas.add_volume(empty)

        try:
            mi, ma = handle.clim
            handle.clim = (min(mi, np.min(data)), max(ma, np.max(data)))
        except Exception as e:
            print("err in clim: ", e)
            handle.clim = (0, 5000)

        print(">>draw:")
        print(f"  data: {data.shape} @ {offset}")
        handle.directly_set_texture_offset(data, offset)
        self._canvas.refresh()

        # # of the chunks will determine the order of the channels in the LUTS
        # # (without additional logic to sort them by index, etc.)
        # if (handles := self._channels.get(ch_key)) is None:
        #     handles = self._create_channel(ch_key)

        # if not handles:
        #     if data.ndim == 2:
        #         handles.append(self._canvas.add_image(data, cmap=handles.cmap))
        #     elif data.ndim == 3:
        #         empty = np.empty((60, 256, 256), dtype=np.uint16)
        #         handles.append(self._canvas.add_volume(empty, cmap=handles.cmap))

        # handles[0].set_data(data, chunk.offset)
        # self._canvas.refresh()


# class NDViewer:
#     def __init__(self, data: Any, state: ViewerState | None) -> None:
#         self._state = state or ViewerState()
#         if data is not None:
#             self.set_data(data)

#     @property
#     def data(self) -> Any:
#         raise NotImplementedError

#     def set_data(self, data: Any) -> None: ...

#     @property
#     def state(self) -> ViewerState:
#         return self._state

#     def set_state(self, state: ViewerState) -> None:
#         # validate...
#         self._state = state

#     def set_visualized_indices(self, indices: tuple[DimKey, DimKey]) -> None:
#         """Set which indices are visualized."""
#         if self._state.channel_index in indices:
#             raise ValueError(
#                 f"channel index ({self._state.channel_index!r}) cannot be in visualized"
#                 f"indices: {indices}"
#             )
#         self._state.visualized_indices = indices
#         self.refresh()

#     def set_channel_index(self, index: DimKey | None) -> None:
#         """Set the channel index."""
#         if index in self._state.visualized_indices:
#             # consider alternatives to raising.
#             # e.g. if len(visualized_indices) == 3, then we could pop index
#             raise ValueError(
#                 f"channel index ({index!r}) cannot be in visualized indices: "
#                 f"{self._state.visualized_indices}"
#             )
#         self._state.channel_index = index
#         self.refresh()

#     def set_current_index(self, index: Mapping[DimKey, Index]) -> None:
#         """Set the currentl displayed index."""
#         self._state.current_index = index
#         self.refresh()

#     def refresh(self) -> None:
#         """Refresh the viewer."""
#         index = self._state.current_index
#         self._chunker.request_index(index)

#     @ensure_main_thread  # type: ignore
#     def _draw_chunk(self, chunk: ChunkResponse) -> None: ...
