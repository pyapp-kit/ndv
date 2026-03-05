# Plan: Spatial Scales & Channel Names

## Context

PRs #165 (scales) and #225 (channel names) need to be reimplemented to fit
the new MVC pattern from PR #233. Both scales and channel names are metadata
that can come from the user (`ArrayDisplayModel`) or the data (`DataWrapper`),
and they need pure resolution in `resolve()`, just like `channel_axis`.

This branch (`channel-scales`) supersedes #165 and #225.

## Design Decisions

1. **Scales** → new `scales: ValidatedEventedDict[AxisKey, float]` on
   `ArrayDisplayModel`. Resolved into `visible_scales: tuple[float, ...]`
   on `ResolvedDisplayState`.

2. **Channel names** → new `channel_names: ValidatedEventedDict[ChannelKey, str]`
   on `ArrayDisplayModel`. Resolved into `channel_names: dict[int, str]`
   on `ResolvedDisplayState`.

3. **DataWrapper** gets two new overridable methods:
   - `channel_names(channel_axis: int | None) -> dict[int, str]`
   - `axis_scales() -> dict[Hashable, float]`
   - Base class: infer scales from uniform coord spacing (works for any
     duck-type with `.coords`), return `{}` for channel_names.
   - Subclasses can override for custom metadata (e.g., attrs, OME).

4. **Resolution priority**: user explicit > data-derived > default.
   - Scales default: `1.0`
   - Channel names: resolver returns only explicitly known names;
     UI fallback to `str(key)` stays in the controller/view layer.
   - Model keys stay in user terms (AxisKey/ChannelKey); normalization
     happens only in `resolve()`.

5. **Scale changes are render-only**: updating scales should update
   transforms + camera + hover mapping, but NOT trigger data refetch.
   `needs_data` in `_apply_changes` remains unchanged.

6. **Slider labels**: Architecture prepares for physical coordinates on
   sliders, but full configurability is a follow-up. `data_coords`
   already carries real coordinate values to the view.

## Implementation

### Phase 1: Model + Resolve (pure, no visual impact)

**`src/ndv/models/_array_display_model.py`**
- Add type aliases:
  ```python
  ScalesMap: TypeAlias = ValidatedEventedDict[AxisKey, float]
  ChannelNamesMap: TypeAlias = ValidatedEventedDict[ChannelKey, str]
  ```
- Add fields:
  ```python
  scales: ScalesMap = Field(default_factory=ScalesMap, frozen=True)
  channel_names: ChannelNamesMap = Field(
      default_factory=ChannelNamesMap, frozen=True
  )
  ```
- Update `ArrayDisplayModelKwargs` TypedDict

**`src/ndv/models/_data_wrapper.py`**
- Add to `DataWrapper` base class:
  ```python
  def channel_names(self, channel_axis: int | None) -> dict[int, str]:
      """Return channel display names from data metadata."""
      return {}

  def axis_scales(self) -> dict[Hashable, float]:
      """Return per-axis scale factors from coordinate spacing."""
      scales: dict[Hashable, float] = {}
      for dim in self.dims:
          coord = self.coords.get(dim)
          if coord is not None and len(coord) >= 2:
              values = [float(v) for v in coord]
              diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
              if all(abs(d - diffs[0]) < 1e-10 for d in diffs):
                  scales[dim] = diffs[0]
      return scales
  ```
  Note: `axis_scales()` on the *base* class infers from coord spacing, so
  any duck-type array with `.coords` (xarray, etc.) benefits automatically.
  Subclasses can override for attrs/metadata.
  Guardrails: skip non-numeric/non-finite coords silently. Allow negative
  scales (descending coords preserve orientation). Only infer from uniform
  spacing.

- Override `channel_names` in `XarrayWrapper` (and potentially base too,
  since `self.coords` is available on all wrappers):
  ```python
  def channel_names(self, channel_axis: int | None) -> dict[int, str]:
      if channel_axis is None:
          return {}
      dim = self.dims[channel_axis]
      coords = self.coords.get(dim, None)
      if coords is None:
          return {}
      return {i: str(v) for i, v in enumerate(coords)}
  ```

**`src/ndv/models/_resolve.py`**
- Add to `ResolvedDisplayState`:
  ```python
  visible_scales: tuple[float, ...]   # aligned with visible_axes
  channel_names: dict[int, str]       # channel_index -> display name
  ```
- Add `_resolve_visible_scales(model, wrapper, visible_axes) -> tuple[float, ...]`:
  For each visible axis: check `model.scales` (normalize key) > `wrapper.axis_scales()` > `1.0`
- Add `_resolve_channel_names(model, wrapper, channel_axis, ...) -> dict[int, str]`:
  For each channel: check `model.channel_names` > `wrapper.channel_names(ch_ax)`.
  Only include channels with an explicit name; omit others so UI falls
  back to `str(key)` in the controller/view layer.
- Update `resolve()` to call both
- Update `EMPTY_STATE` with `visible_scales=()`, `channel_names={}`
- Include `visible_scales` in `__eq__` (scale change → re-render)
- Exclude `channel_names` from `__eq__` (cosmetic, no data refetch)

### Phase 2: Channel Names (wire to views)

**`src/ndv/controllers/_array_viewer.py`**
- In `_set_model_connected`: connect `model.channel_names.value_changed` → `_re_resolve`
- Add `_update_channel_names(names: dict[int, str])`:
  ```python
  def _update_channel_names(self, names: dict[int, str]) -> None:
      for key, ctrl in self._lut_controllers.items():
          name = names.get(key, "" if key is None else str(key))
          for view in ctrl.lut_views:
              view.set_channel_name(name)
  ```
- In `_apply_changes`: if `old.channel_names != new.channel_names`, call it
- In `_fully_synchronize_view`: call after `lut_ctr.synchronize()`
- **First-paint correctness**: In `_on_data_response_ready`, after creating
  a new `ChannelController`, immediately apply the resolved name from
  `self._resolved.channel_names`. Otherwise new controllers show raw
  index keys until the next re-resolve.

### Phase 3: Scales (wire to canvas — both backends)

**`src/ndv/views/bases/_graphics/_canvas.py`**
- Add non-abstract method to `ArrayCanvas`:
  ```python
  def set_scales(self, scales: tuple[float, ...]) -> None:
      """Set per-visible-axis scale factors for rendering."""
  ```

**`src/ndv/views/_pygfx/_array_canvas.py`**
- Implement `set_scales`: apply `.local.scale_x/y/z` on image/volume nodes
- Call `set_range()` after to adjust camera

**`src/ndv/views/_vispy/_array_canvas.py`**
- Implement `set_scales`: apply `STTransform(scale=...)` on scene nodes
- Call `set_range()` after

**`src/ndv/controllers/_array_viewer.py`**
- In `_set_model_connected`: connect `model.scales.value_changed` → `_re_resolve`
- In `_apply_changes`: if `old.visible_scales != new.visible_scales`,
  call `self._canvas.set_scales(new.visible_scales)`
- In `_fully_synchronize_view`: call `self._canvas.set_scales(...)`
- In `_get_values_at_world_point`: map world coordinates back to data
  pixel indices using resolved visible-axis scales. Use the resolved
  visible-axis order explicitly (not just scales[0]/scales[1] blindly),
  with backend/world XY→data-axis mapping handled correctly.

### Axis ordering note
Data is laid out slowest-to-fastest (e.g., ZYX for 3D). `visible_axes` and
`visible_scales` follow the same order. Graphics frameworks use XYZ. The
canvas backends must reverse the order when applying transforms.

### Channel key domain in `_update_channel_names`
Controllers can have non-int keys (`None` for default/grayscale, `"RGB"`
for RGBA mode). Resolved `channel_names: dict[int, str]` only covers
integer channel indices. `_update_channel_names` must handle non-int keys
intentionally: `None` → `""`, `"RGB"` → `"RGB"`, int keys → resolved name
or fallback to `str(key)`.

## Verification

1. Unit tests for `_resolve_visible_scales` and `_resolve_channel_names`
   in `tests/test_models.py` or similar
2. Test user scales override data-derived scales
3. Test user channel_names override data-derived names
4. Test xarray data provides both scales (from coord spacing) and
   channel names (from channel dim coords) automatically
5. Test `set_scales` is called on canvas in `_apply_changes`
6. Test channel names appear in LUT views via `set_channel_name`
7. Test `axis_scales()` with nan/inf coords → silently skipped, no scale inferred
8. Test `axis_scales()` with descending coords → negative scale returned
9. Manual smoke test:
   ```python
   import ndv, numpy as np
   ndv.imshow(np.random.rand(3, 100, 200),
              channel_axis=0, channel_mode="composite",
              channel_names={0: "DAPI", 1: "GFP", 2: "mCherry"},
              scales={1: 0.5, 2: 0.1})
   ```

## Key Files
- `src/ndv/models/_array_display_model.py` — new fields
- `src/ndv/models/_data_wrapper.py` — new methods + base coord-spacing logic
- `src/ndv/models/_resolve.py` — resolution functions + ResolvedDisplayState
- `src/ndv/controllers/_array_viewer.py` — wire to canvas + LUT views
- `src/ndv/views/bases/_graphics/_canvas.py` — `set_scales` API
- `src/ndv/views/_pygfx/_array_canvas.py` — pygfx implementation
- `src/ndv/views/_vispy/_array_canvas.py` — vispy implementation
