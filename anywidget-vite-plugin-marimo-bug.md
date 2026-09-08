# @anywidget/vite plugin breaks Lit classMap in marimo's Declarative Shadow DOM

## Summary

The `@anywidget/vite` plugin causes a runtime error when Lit-based web components
(specifically Web Awesome / Shoelace components that use `classMap()`) are rendered
inside marimo's Declarative Shadow DOM (DSD) environment. Building without the
plugin produces a working bundle.

## Error

```
Uncaught (in promise) Error: `classMap()` can only be used in the `class` attribute
and must be the only part in the attribute.
```

This error comes from Lit's `classMap` directive, which is used internally by Web
Awesome components (`wa-button`, `wa-select`, etc.). The directive validates that
it's being used in a `class` attribute binding, but something about the plugin's
transform causes it to misidentify the binding context.

## Reproduction

1. Create a vite-bundled anywidget ESM that imports Lit components and Web Awesome:

```js
import { LitElement, html } from "lit";
import "@awesome.me/webawesome/dist/components/button/button.js";

class MyWidget extends LitElement {
  createRenderRoot() { return this; }
  render() {
    return html`<wa-button size="small">Click</wa-button>`;
  }
}
customElements.define("my-widget", MyWidget);

function render({ model, el }) {
  el.appendChild(document.createElement("my-widget"));
}
export default { render };
```

2. Build with the plugin:

```js
// vite.config.js
import anywidget from "@anywidget/vite";
export default defineConfig({
  plugins: [anywidget()],
  // ...
});
```

3. Use the built ESM as an `anywidget.AnyWidget._esm` in marimo.

4. The `classMap()` error fires and the component fails to render.

5. Rebuild **without** the plugin:

```js
// vite.config.js — no plugins
export default defineConfig({
  build: { /* same config minus plugins */ },
});
```

6. The component renders correctly.

## Environment

- `@anywidget/vite`: (version bundled with ndv, via npm)
- `lit`: 3.x (bundled via Web Awesome)
- `marimo`: 0.22.0
- Web Awesome: latest as of 2026-03

## Analysis

Marimo renders anywidgets inside a Declarative Shadow DOM (`<template
shadowrootmode="open">`). The `@anywidget/vite` plugin transforms the ESM for
blob URL loading compatibility. This transform appears to alter how Lit's tagged
template literals are bundled, causing the `classMap` directive to lose its
attribute-type context when Lit hydrates inside the DSD.

Without the plugin, vite's standard ES module bundling preserves Lit's template
tag functions correctly, and `classMap` works as expected in the DSD context.

The issue does NOT occur in Jupyter, which renders widgets in a normal DOM context
(no DSD). It is specific to the combination of:
- `@anywidget/vite` plugin transform
- Lit's `classMap` directive (used internally by Web Awesome)
- Marimo's Declarative Shadow DOM rendering

## Workaround

Build without the `@anywidget/vite` plugin. The standard vite ESM build with
`inlineDynamicImports: true` works correctly for marimo.

## Related

- Marimo anywidget renderer: `marimo/_plugins/ui/_impl/from_anywidget.py`
- Lit classMap docs: https://lit.dev/docs/templates/directives/#classmap
