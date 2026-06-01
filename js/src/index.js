// Make customElements.define idempotent — anywidget may re-evaluate this
// ESM blob on cell re-run, causing Web Awesome to re-register elements.
const _origDefine = customElements.define.bind(customElements);
customElements.define = (name, ctor, options) => {
  if (!customElements.get(name)) _origDefine(name, ctor, options);
};

// Web Awesome theme + base styles
import "@awesome.me/webawesome/dist/styles/themes/default.css";
import "@awesome.me/webawesome/dist/styles/webawesome.css";
import "./styles.css";

// Web Awesome components
import "@awesome.me/webawesome/dist/components/slider/slider.js";
import "@awesome.me/webawesome/dist/components/select/select.js";
import "@awesome.me/webawesome/dist/components/option/option.js";
import "@awesome.me/webawesome/dist/components/checkbox/checkbox.js";
import "@awesome.me/webawesome/dist/components/button/button.js";
import "@awesome.me/webawesome/dist/components/tooltip/tooltip.js";
import "@awesome.me/webawesome/dist/components/dropdown/dropdown.js";
import "@awesome.me/webawesome/dist/components/icon/icon.js";

// ndv Lit components
import "./components/ndv-viewer.js";
import { getThemeInfo, watchThemeChanges } from "./theme.js";

// Inject CSS into document.head so :root custom properties (used by
// Web Awesome theme) cascade into shadow DOMs.  Without this, the
// theme vars defined under :root won't reach child shadow roots when
// the widget itself is rendered inside a shadow root (e.g. marimo).
let _headStyleInjected = false;
function _ensureHeadStyles(model, info) {
  if (_headStyleInjected) return;
  _headStyleInjected = true;

  // Override white backgrounds as early as possible to minimize flash —
  // but only in VSCode, which is the only environment that needs it.
  if (info.environment === "vscode") overrideVscodeWhiteBackgrounds();

  const css = model.get("_css");
  if (!css) return;
  const style = document.createElement("style");
  style.setAttribute("data-ndv-style", "");
  style.textContent = css;
  document.head.appendChild(style);
}

/**
 * VSCode-only background override.
 *
 * VSCode injects `.cell-output-ipywidget-background { background: white !important }`
 * via a dynamically created <style> tag, and the WA theme sets `color-scheme: light`
 * on :root which paints the webview canvas white. We override both using inline
 * styles with !important. Do NOT call this outside VSCode — mutating documentElement
 * and body styles is a global side effect that leaks into the host page.
 */
function overrideVscodeWhiteBackgrounds() {
  const bg =
    getComputedStyle(document.documentElement)
      .getPropertyValue("--vscode-editor-background")
      .trim() || "transparent";

  for (const el of document.querySelectorAll(".cell-output-ipywidget-background")) {
    el.style.setProperty("background", bg, "important");
  }
  document.documentElement.style.setProperty("background-color", bg, "important");
  document.body.style.setProperty("background-color", bg, "important");
}

/** Apply theme to a viewer element and sync state to the Python model. */
function applyTheme(viewer, model, info) {
  const isDark = info.kind === "dark" || info.kind === "high-contrast-dark";
  // Localized: only affects descendants of the viewer, not the host page.
  viewer.classList.toggle("wa-dark", isDark);

  // VSCode-specific hacks. In other environments we keep the widget's default
  // background transparent and leave <html>/<body> alone so the host page's
  // own theme is fully in control.
  if (info.environment === "vscode") {
    document.documentElement.style.setProperty(
      "color-scheme",
      isDark ? "dark" : "light",
      "important",
    );
    overrideVscodeWhiteBackgrounds();
  }

  // Defer model sync to avoid deadlock — VSCode's webview can block if
  // save_changes() is called synchronously during render().
  queueMicrotask(() => {
    model.set("_theme_kind", info.kind);
    model.set("_theme_background", info.background || "");
    model.save_changes();
  });
}

/** @param {{ model: any, el: HTMLElement }} ctx */
function render({ model, el }) {
  // Detect theme up-front so style injection can branch on environment.
  const info = getThemeInfo();

  _ensureHeadStyles(model, info);

  const viewer = document.createElement("ndv-viewer");
  viewer.model = model;
  el.appendChild(viewer);

  applyTheme(viewer, model, info);

  // In VSCode, re-run the override later — the .cell-output-ipywidget-background
  // container may not exist at first render.
  if (info.environment === "vscode") {
    setTimeout(() => overrideVscodeWhiteBackgrounds(), 200);
  }

  // Watch for runtime theme switches (e.g. VSCode dark → light, OS light → dark)
  const stopWatching = watchThemeChanges((newInfo) =>
    applyTheme(viewer, model, newInfo),
  );

  return () => {
    stopWatching();
    viewer.remove();
  };
}

// AFM spec requires default export — named `export function render` is deprecated.
// See: https://anywidget.dev/en/afm/
export default { render };
