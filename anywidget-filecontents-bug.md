# Bug: `FileContents` JSON serialization failure when `_esm` and `_css` are both file paths

## Summary

When using `MimeBundleDescriptor` (or the `@widget` decorator) with **both** `_esm` and `_css` as file paths, the initial comm open fails with:

```
Can't clean for JSON: <anywidget._file_contents.FileContents object at 0x...>
```

The widget falls back to its `__repr__` instead of rendering the JS frontend.

## How we found it

We were building a custom widget using `@widget` + `@dataclass` + `@psygnal.evented`:

```python
from pathlib import Path
from dataclasses import dataclass
from anywidget.experimental import widget

@widget(
    esm=Path(__file__).parent / "static" / "index.js",
    css=Path(__file__).parent / "static" / "style.css",
)
@dataclass
class MyWidget:
    value: int = 0
```

Displaying this in a Jupyter notebook produced the warning:

```
UserWarning: Error in Anywidget repr:
Can't clean for JSON: <anywidget._file_contents.FileContents object at 0x...>
```

and showed the dataclass repr instead of the widget.

Our initial assumption was that our custom `_get_anywidget_state` method was interfering with serialization. We removed it entirely, but the error persisted. We then traced through the descriptor code and found the root cause.

## Root cause

In `anywidget/_descriptor.py`, `ReprMimeBundle.__init__` (around line 330):

```python
for key, value in self._extra_state.items():
    if isinstance(value, (VirtualFileContents, FileContents)):
        self._extra_state[key] = str(value)  # Convert to string

        @value.changed.connect
        def _on_change(new_contents: str, key: str = key) -> None:
            self._extra_state[key] = new_contents
            self.send_state(key)

    self._comm = _get_or_create_comm(          # <-- INSIDE the for loop
        obj=obj,
        get_state=lambda: {
            **self._get_state(obj, include=None),
            **self._extra_state,                # <-- captures dict by reference
        },
    )
```

The `self._comm = _get_or_create_comm(...)` call is **inside** the `for` loop (same indentation level as the `if` block). On the **first iteration** (e.g., `_esm`):

1. `_esm` is converted from `FileContents` → `str` ✓
2. `_get_or_create_comm` is called, which calls `get_state()` immediately (line 120: `open_comm(initial_state=get_state())`)
3. `get_state()` spreads `**self._extra_state` into the state dict
4. But `_css` **hasn't been processed yet** — it's still a `FileContents` object
5. The comm tries to JSON-serialize the state → fails on the `FileContents` object

When only `_esm` is provided (no `_css`), the loop has one iteration and the bug doesn't manifest. It only fails when there are **two or more** `FileContents` values in `_extra_state`.

## Fix

The `_get_or_create_comm` call should be moved **outside** the `for` loop:

```python
# Convert all FileContents FIRST
for key, value in self._extra_state.items():
    if isinstance(value, (VirtualFileContents, FileContents)):
        self._extra_state[key] = str(value)

        @value.changed.connect
        def _on_change(new_contents: str, key: str = key) -> None:
            self._extra_state[key] = new_contents
            self.send_state(key)

# THEN create the comm (after all values are serializable)
self._comm = _get_or_create_comm(
    obj=obj,
    get_state=lambda: {
        **self._get_state(obj, include=None),
        **self._extra_state,
    },
)
```

## Workaround

Pass file contents as strings instead of `Path` objects, so they never become `FileContents`:

```python
_STATIC = Path(__file__).parent / "static"

@widget(
    esm=(_STATIC / "index.js").read_text(encoding="utf-8"),
    css=(_STATIC / "style.css").read_text(encoding="utf-8"),
)
@dataclass
class MyWidget:
    value: int = 0
```

This bypasses `try_file_contents` entirely (strings are not wrapped in `FileContents`), but loses file-watching/HMR during development.

## Affected version

Verified in anywidget 0.9.21 and the current `main` branch (as of 2026-03-31).
