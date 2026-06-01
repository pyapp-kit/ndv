---
name: marimo_reload_bug
description: Anywidget descriptor-based widgets stop updating after page reload in marimo run mode
type: project
---

## Bug: Descriptor-based anywidgets don't reconnect on page reload

**Affects:** `marimo run` (and likely `marimo edit`) with anywidget `@widget` descriptor API

**Symptoms:** On first page load, all widgets work (canvas renders, controls respond).
After browser reload, the cell re-executes (confirmed via print), new Python widget
objects are created, the JS `render()` function is called with the new model (confirmed
via console.log — model has correct state), but neither the pygfx canvas nor the
NdvWidgetState controls widget respond to interaction or update.

**What we confirmed:**
- Python cell re-executes on reload (new objects created)
- JS `render()` is called on reload with correct model data (sliders: 4)
- JS `_bindModel` fires (even fires twice on reload, once on first load)
- The model has the correct state — it's the comm/sync that's broken

**Likely cause:** The `_repr_mimebundle_` → `_maybe_as_anywidget_html` path in
`repr_formatters.py` creates a `<marimo-anywidget>` element with a `model_id`.
On session reconnect, the frontend re-renders the HTML and calls `render()` with
a model from `MODEL_MANAGER`. But the new Python-side comm (created by the
descriptor during cell re-execution) may not properly replace the stale comm
in marimo's comm registry, so state updates from Python never reach the frontend.

The `mo.ui.anywidget()` path (used for traditional AnyWidget subclasses like
the pygfx canvas) likely has the same issue since it also breaks on reload.

**To reproduce:**
1. `NDV_CANVAS_BACKEND=pygfx uv run marimo run marimo_example.py`
2. Interact with the widget (works)
3. Reload the browser page
4. Widget appears but is frozen — no interaction works

**Where to report:** https://github.com/marimo-team/marimo/issues — relates to
the descriptor API support from https://github.com/marimo-team/marimo/pull/8972.
The `mo.ui.anywidget()` reload path may also be affected.

**Workaround:** None currently. Restarting the marimo server works.
