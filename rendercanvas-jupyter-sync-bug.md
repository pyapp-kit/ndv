# Bug: `JupyterRenderCanvas.get_frame()` returns previous frame (async readback)

## Summary

`JupyterRenderCanvas.get_frame()` returns stale frames because `_time_to_draw()` uses `force_sync=False`, allowing the GPU readback to complete asynchronously. By the time `get_frame()` returns `self._last_image`, it still holds the *previous* frame's data.

Every interaction appears to be "behind" by one frame.

## Root Cause

In `rendercanvas/jupyter.py`:

```python
def get_frame(self):
    self._time_to_draw()
    return self._last_image
```

`_time_to_draw()` calls `_draw_and_present(force_sync=False)` (base.py line 532). With `force_sync=False`, the readback path in `WgpuContextToBitmap._rc_present()` is:

```python
# wgpucontext.py line 330-332
awaitable = self._downloader.initiate_download(self._texture, self._present_params)
return {"method": "async", "awaitable": awaitable}
```

This initiates an async download and returns immediately. The callback `_finish_present()` updates `self._last_image` *later* — but `get_frame()` has already returned the old value.

## Correct Behavior (offscreen backend)

The offscreen backend's `draw()` method correctly uses synchronous readback:

```python
# offscreen.py
def draw(self):
    self.force_draw()  # calls _draw_and_present(force_sync=True)
    return self._last_image
```

With `force_sync=True`, `_rc_present()` calls `do_sync_download()` which blocks until the GPU readback completes before returning.

## Suggested Fix

`JupyterRenderCanvas.get_frame()` should ensure synchronous readback, either by:

1. Calling `force_draw()` instead of `_time_to_draw()`:
   ```python
   def get_frame(self):
       self.force_draw()
       return self._last_image
   ```

2. Or overriding `_draw_and_present` to use `force_sync=True` when called from the Jupyter context.

## Workaround

Subclass and override `get_frame()`:

```python
from rendercanvas.jupyter import JupyterRenderCanvas

class SyncJupyterRenderCanvas(JupyterRenderCanvas):
    def get_frame(self):
        self._draw_and_present(force_sync=True)
        return self._last_image
```

## Affected Version

rendercanvas 2.6.3
