"""Resolve display model + data wrapper into a deterministic display state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from ._array_display_model import ChannelMode

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    import numpy as np

    from ndv._types import ChannelKey

    from ._array_display_model import ArrayDisplayModel
    from ._data_wrapper import DataWrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataRequest:
    """Request object for data slicing."""

    wrapper: DataWrapper = field(repr=False)
    index: Mapping[int, int | slice]
    visible_axes: tuple[int, ...]
    channel_axis: int | None
    channel_mode: ChannelMode


@dataclass(frozen=True, slots=True)
class DataResponse:
    """Response object for data requests."""

    n_visible_axes: int
    data: Mapping[ChannelKey, np.ndarray] = field(repr=False)
    request: DataRequest | None = None


@dataclass(frozen=True, slots=True)
class ResolvedDisplayState:
    """Frozen snapshot of resolved display state (ArrayDisplayModel + DataWrapper).

    At any given time, the viewer's display state is influenced by both the current
    ArrayDisplayModel (e.g. which axes are visible, channel mode, current_index), which
    is mutable and can be updated by the user, and the DataWrapper, which has awareness
    of the underlying data (e.g. dimension names, coordinates, metadata).

    The term "resolution" here refers to the process of taking the user's "intentions",
    as expressed in the `ArrayDisplayModel`, and *resolving* them against the reality of
    the data, as represented by the DataWrapper.  This includes normalizing axis keys to
    positive integers, injecting values derived from the data (e.g. guessing a channel
    axis or inferring coordinates), and computing derived state.

    The output of that resolution process is this `ResolvedDisplayState`, which is a
    deterministic, immutable, hashable snapshot of the display state that can be used
    for diffing and driving the actual data slicing and visualization logic.

    Produced by `resolve()`. Used for diffing in `ArrayViewer._apply_changes()`.
    """

    visible_axes: tuple[int, ...]
    channel_axis: int | None
    channel_mode: ChannelMode
    current_index: dict[int, int | slice]
    data_coords: dict[int, tuple]
    hidden_sliders: frozenset[Hashable]
    summary_info: str
    visible_scales: tuple[float, ...]
    channel_names: dict[int, str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResolvedDisplayState):
            return NotImplemented
        return (
            self.visible_axes == other.visible_axes
            and self.channel_axis == other.channel_axis
            and self.channel_mode == other.channel_mode
            and self.current_index == other.current_index
            and self.data_coords == other.data_coords
            and self.hidden_sliders == other.hidden_sliders
            and self.visible_scales == other.visible_scales
            # summary_info excluded: metadata-only, should not trigger data fetch
            # channel_names excluded: cosmetic-only, should not trigger data fetch
        )

    def __hash__(self) -> int:
        return id(self)

    def __rich_repr__(self) -> Iterable[tuple[str, object]]:
        for _field in fields(self):
            if _field.name == "data_coords":
                # return truncated data_coords for readability
                yield (
                    _field.name,
                    {
                        k: v if len(v) <= 5 else ((*v[:2], "...", *v[-2:]))
                        for k, v in getattr(self, _field.name).items()
                    },
                )
            else:
                yield _field.name, getattr(self, _field.name)


EMPTY_STATE = ResolvedDisplayState(
    visible_axes=(),
    channel_axis=None,
    channel_mode=ChannelMode.GRAYSCALE,
    current_index={},
    data_coords={},
    hidden_sliders=frozenset(),
    summary_info="",
    visible_scales=(),
    channel_names={},
)


def _norm_visible_axes(
    model: ArrayDisplayModel, wrapper: DataWrapper
) -> tuple[int, ...]:
    """Return visible_axes as positive integers.

    Falls back to the last N dims if any axis is stale.
    """
    try:
        return tuple(wrapper.normalize_axis_key(ax) for ax in model.visible_axes)
    except (IndexError, KeyError):
        n = min(len(model.visible_axes), len(wrapper.dims))
        return tuple(range(len(wrapper.dims) - n, len(wrapper.dims)))


def _norm_channel_axis(model: ArrayDisplayModel, wrapper: DataWrapper) -> int | None:
    """Return channel_axis as a positive integer, or None.

    If the model has no channel_axis but is in a multichannel mode,
    guess one from the wrapper (avoiding visible axes).
    """
    if model.channel_axis is not None:
        try:
            return wrapper.normalize_axis_key(model.channel_axis)
        except (IndexError, KeyError):
            return None  # stale channel_axis from previous data?

    guess = wrapper.guess_channel_axis()
    if guess is None:
        return None

    try:
        normed_guess = wrapper.normalize_axis_key(guess)
    except Exception:
        return None

    # don't use a visible axis as the channel axis
    normed_vis = _norm_visible_axes(model, wrapper)
    if normed_guess in normed_vis:
        return None

    return normed_guess


def _norm_current_index(
    model: ArrayDisplayModel, wrapper: DataWrapper
) -> dict[int, int | slice]:
    """Return current_index with positive integer axis keys.

    Handles duplicate keys (e.g. both 'Z' and 0 mapping to the same axis)
    by giving priority to non-integer keys and logging a warning.
    Does NOT mutate the model.
    """
    output: dict[int, int | slice] = {}
    source_key: dict[int, Hashable] = {}  # tracks which original key wrote each entry

    for key, val in model.current_index.items():
        try:
            normed = wrapper.normalize_axis_key(key)
        except (IndexError, KeyError):
            continue  # stale key from previous data
        if normed in output:
            # prefer named keys (e.g. "Z") over raw integer keys
            is_raw_int = normed == key
            winner = source_key[normed] if is_raw_int else key
            logger.warning(
                "Axis key %r normalized to %r, which is also in current_index. "
                "Using %r value.",
                winner,
                normed,
                winner,
            )
            if is_raw_int:
                continue

        output[normed] = val
        source_key[normed] = key

    return output


def _norm_data_coords(wrapper: DataWrapper) -> dict[int, tuple]:
    """Return data coordinates as {positive_int: tuple_of_values}."""
    return {
        wrapper.normalize_axis_key(d): tuple(wrapper.coords[d]) for d in wrapper.dims
    }


def _resolve_visible_scales(
    model: ArrayDisplayModel,
    wrapper: DataWrapper,
    visible_axes: tuple[int, ...],
) -> tuple[float, ...]:
    """Return scale factors aligned with visible_axes.

    Priority: user explicit (model.scales) > data-derived > default (1.0).
    """
    data_scales = wrapper.axis_scales()
    scales: list[float] = []
    for ax in visible_axes:
        # check user model.scales (try both positive int and dim label)
        user_scale = None
        for key in model.scales:
            try:
                if wrapper.normalize_axis_key(key) == ax:
                    user_scale = model.scales[key]
                    break
            except (IndexError, KeyError):
                continue
        if user_scale is not None:
            scales.append(user_scale)
        else:
            # try data-derived: axis_scales uses dim labels as keys
            dim_label = wrapper.dims[ax]
            if dim_label in data_scales:
                scales.append(data_scales[dim_label])
            elif ax in data_scales:
                scales.append(data_scales[ax])
            else:
                scales.append(1.0)
    return tuple(scales)


def _resolve_channel_names(
    model: ArrayDisplayModel,
    wrapper: DataWrapper,
    channel_axis: int | None,
) -> dict[int, str]:
    """Return channel display names.

    Priority: user explicit (model.channel_names) > data-derived.
    Only includes channels with an explicit name.
    """
    data_names = wrapper.channel_names(channel_axis)
    result: dict[int, str] = {}
    # merge data-derived names first, then override with user names
    result.update(data_names)
    for key, name in model.channel_names.items():
        if isinstance(key, int):
            result[key] = name
    return result


def _compute_hidden_sliders(
    visible_axes: tuple[int, ...],
    channel_axis: int | None,
    channel_mode: ChannelMode,
    data_coords: dict[int, tuple],
    wrapper: DataWrapper,
) -> frozenset[Hashable]:
    """Compute the set of slider keys to hide."""
    hidden: set[int] = set(visible_axes)
    if channel_mode.is_multichannel() and channel_axis is not None:
        hidden.add(channel_axis)
    # hide singleton axes
    hidden.update(ax for ax, coord in data_coords.items() if len(coord) < 2)
    # include dim names so sliders hide regardless of key form
    return frozenset(hidden | {wrapper.dims[ax] for ax in hidden})


def resolve(model: ArrayDisplayModel, wrapper: DataWrapper) -> ResolvedDisplayState:
    """Pure function: resolve model + wrapper into a frozen display state.

    Does NOT mutate the model.
    """
    visible_axes = _norm_visible_axes(model, wrapper)
    channel_axis = _norm_channel_axis(model, wrapper)
    current_index = _norm_current_index(model, wrapper)
    data_coords = _norm_data_coords(wrapper)

    hidden_sliders = _compute_hidden_sliders(
        visible_axes, channel_axis, model.channel_mode, data_coords, wrapper
    )
    visible_scales = _resolve_visible_scales(model, wrapper, visible_axes)
    channel_names = _resolve_channel_names(model, wrapper, channel_axis)

    return ResolvedDisplayState(
        visible_axes=visible_axes,
        channel_axis=channel_axis,
        channel_mode=model.channel_mode,
        current_index=current_index,
        data_coords=data_coords,
        hidden_sliders=hidden_sliders,
        summary_info=wrapper.summary_info(),
        visible_scales=visible_scales,
        channel_names=channel_names,
    )


def build_slice_requests(
    resolved: ResolvedDisplayState, wrapper: DataWrapper
) -> list[DataRequest]:
    """Build data slice requests from resolved state."""
    requested_slice: dict[int, int | slice] = dict(resolved.current_index)

    for ax in resolved.visible_axes:
        if not isinstance(requested_slice.get(ax), slice):
            requested_slice[ax] = slice(None)

    c_ax = resolved.channel_axis
    if c_ax is not None:
        if resolved.channel_mode.is_multichannel():
            if not isinstance(requested_slice.get(c_ax), slice):
                requested_slice[c_ax] = slice(None)
        else:
            # in non-multichannel mode with channel_axis set,
            # we still want the default_lut behavior
            c_ax = None

    # convert all int indices to single-element slices to preserve dimensions
    for ax, val in requested_slice.items():
        if isinstance(val, int):
            requested_slice[ax] = slice(val, val + 1)

    return [
        DataRequest(
            wrapper=wrapper,
            index=requested_slice,
            visible_axes=resolved.visible_axes,
            channel_axis=c_ax,
            channel_mode=resolved.channel_mode,
        )
    ]


def process_request(req: DataRequest) -> DataResponse:
    """Process a data request and return the sliced data as a DataResponse."""
    data = req.wrapper.isel(req.index)

    vis_ax = req.visible_axes
    t_dims = vis_ax + tuple(i for i in range(data.ndim) if i not in vis_ax)

    data_response: dict[ChannelKey, np.ndarray] = {}
    ch_ax = req.channel_axis

    if req.channel_mode == ChannelMode.RGBA:
        data_response["RGB"] = data.transpose(*t_dims).squeeze()
    elif ch_ax is None:
        data_response[None] = data.transpose(*t_dims).squeeze()
    else:
        for i in range(data.shape[ch_ax]):
            ch_keepdims = (slice(None),) * ch_ax + (i,) + (None,)
            ch_data = data[ch_keepdims]
            data_response[i] = ch_data.transpose(*t_dims).squeeze()

    return DataResponse(n_visible_axes=len(vis_ax), data=data_response, request=req)
