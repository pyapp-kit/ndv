/**
 * Theme detection for notebook environments (VSCode, JupyterLab, Colab).
 */

/**
 * Detect current theme and environment.
 * @returns {{ kind: string, background: string|null, foreground: string|null, environment: string }}
 */
export function getThemeInfo() {
  const result = {
    kind: "dark",
    background: null,
    foreground: null,
    environment: "unknown",
  };

  const body = document.body;
  const bodyClasses = body.classList;

  // VSCode detection (body classes set by the notebook webview)
  if (bodyClasses.contains("vscode-light")) {
    result.kind = "light";
    result.environment = "vscode";
  } else if (bodyClasses.contains("vscode-dark")) {
    result.kind = "dark";
    result.environment = "vscode";
  } else if (bodyClasses.contains("vscode-high-contrast-light")) {
    result.kind = "light";
    result.environment = "vscode";
  } else if (bodyClasses.contains("vscode-high-contrast")) {
    result.kind = "dark";
    result.environment = "vscode";
  }
  // JupyterLab detection
  else if (body.dataset.jpThemeLight !== undefined) {
    result.kind = body.dataset.jpThemeLight === "true" ? "light" : "dark";
    result.environment = "jupyterlab";
  }
  // Colab detection
  else if (bodyClasses.contains("dark")) {
    result.kind = "dark";
    result.environment = "colab";
  }
  // Fallback: try body bg luminance, then system preference. Leave
  // environment as "unknown" so callers know not to apply host-specific
  // hacks (e.g. VSCode background overrides).
  else {
    const bg = getComputedStyle(body).backgroundColor;
    const darkFromBg = isColorDark(bg);
    if (darkFromBg !== null) {
      result.kind = darkFromBg ? "dark" : "light";
    } else if (
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ) {
      result.kind = "dark";
    } else {
      result.kind = "light";
    }
  }

  // Read CSS variables when available
  const style = getComputedStyle(document.documentElement);

  const vscodeBg = style.getPropertyValue("--vscode-editor-background").trim();
  const vscodeFg = style.getPropertyValue("--vscode-editor-foreground").trim();
  if (vscodeBg) result.background = vscodeBg;
  if (vscodeFg) result.foreground = vscodeFg;

  // JupyterLab fallback
  if (!result.background) {
    const jpBg = style.getPropertyValue("--jp-layout-color0").trim();
    if (jpBg) result.background = jpBg;
  }

  return result;
}

/**
 * Watch for theme changes and call `callback(getThemeInfo())` when detected.
 * Returns a cleanup function that disconnects the observer.
 * @param {(info: ReturnType<typeof getThemeInfo>) => void} callback
 * @returns {() => void}
 */
export function watchThemeChanges(callback) {
  let timer = null;
  const debounced = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      callback(getThemeInfo());
    }, 100);
  };

  const observer = new MutationObserver(debounced);

  // VSCode changes body class on theme switch; JupyterLab changes data attrs
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["class", "data-jp-theme-light", "data-jp-theme-name"],
  });

  // VSCode injects theme CSS variables as inline styles on <html>
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["style"],
  });

  // System-level light/dark preference (used as the fallback in unknown envs)
  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  mq?.addEventListener?.("change", debounced);

  return () => {
    if (timer) clearTimeout(timer);
    observer.disconnect();
    mq?.removeEventListener?.("change", debounced);
  };
}

/**
 * Parse an rgb/rgba color string and return whether it's dark.
 * Returns `null` if the color is fully transparent or unparsable —
 * callers should fall back to another signal rather than guessing.
 */
function isColorDark(color) {
  if (!color || color === "transparent") return null;
  const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?/);
  if (!match) return null;
  const [, r, g, b, a] = match;
  if (a !== undefined && Number(a) === 0) return null;
  const luminance = (0.299 * Number(r) + 0.587 * Number(g) + 0.114 * Number(b)) / 255;
  return luminance < 0.5;
}
