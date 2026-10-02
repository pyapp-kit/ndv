import { html, LitElement } from "lit";
import "./ndv-dim-sliders.js";
import "./ndv-lut-panel.js";
import "./ndv-toolbar.js";

export class NdvViewer extends LitElement {
  // Use Light DOM so Web Awesome components get proper CSS
  createRenderRoot() {
    return this;
  }

  static properties = {
    model: { type: Object },
    _dataInfo: { state: true },
    _hoverInfo: { state: true },
    _sliders: { state: true },
    _luts: { state: true },
    _channelMode: { state: true },
    _channelModeOptions: { state: true },
    _showControls: { state: true },
    _showDataInfo: { state: true },
    _progressVisible: { state: true },
    _is3d: { state: true },
    _show3dButton: { state: true },
    _showChannelModeSelector: { state: true },
    _showResetZoomButton: { state: true },
    _showRoiButton: { state: true },
    _showHistogramButton: { state: true },
    _useSharedHistogram: { state: true },
    _sharedHistogramVisible: { state: true },
    _sharedHistogramLog: { state: true },
  };

  constructor() {
    super();
    this.model = null;
    this._dataInfo = "";
    this._hoverInfo = "";
    this._sliders = [];
    this._luts = [];
    this._channelMode = "grayscale";
    this._channelModeOptions = [];
    this._showControls = true;
    this._showDataInfo = true;
    this._progressVisible = false;
    this._is3d = false;
    this._show3dButton = true;
    this._showChannelModeSelector = true;
    this._showResetZoomButton = true;
    this._showRoiButton = false;
    this._showHistogramButton = true;
    this._useSharedHistogram = false;
    this._sharedHistogramVisible = false;
    this._sharedHistogramLog = false;
    this._cleanups = [];
  }

  connectedCallback() {
    super.connectedCallback();
    if (this.model) this._bindModel();
  }

  updated(changed) {
    if (changed.has("model") && this.model) this._bindModel();
  }

  _bindModel() {
    // Clean up any previous bindings before re-binding
    for (const cleanup of this._cleanups) cleanup();
    this._cleanups = [];

    const m = this.model;
    const sync = (field, prop) => {
      this[prop] = m.get(field) ?? this[prop];
      const handler = () => {
        this[prop] = m.get(field);
      };
      m.on(`change:${field}`, handler);
      this._cleanups.push(() => m.off(`change:${field}`, handler));
    };

    sync("data_info", "_dataInfo");
    sync("hover_info", "_hoverInfo");
    sync("sliders", "_sliders");
    sync("luts", "_luts");
    sync("channel_mode", "_channelMode");
    sync("channel_mode_options", "_channelModeOptions");
    sync("show_controls", "_showControls");
    sync("show_data_info", "_showDataInfo");
    sync("progress_visible", "_progressVisible");
    sync("is_3d", "_is3d");
    sync("show_3d_button", "_show3dButton");
    sync("show_channel_mode_selector", "_showChannelModeSelector");
    sync("show_reset_zoom_button", "_showResetZoomButton");
    sync("show_roi_button", "_showRoiButton");
    sync("show_histogram_button", "_showHistogramButton");
    sync("use_shared_histogram", "_useSharedHistogram");
    sync("shared_histogram_visible", "_sharedHistogramVisible");
    sync("shared_histogram_log", "_sharedHistogramLog");

    // Directly toggle .ndv-hist-wrap elements when histogram visibility
    // changes (used by marimo where the histogram is a sibling widget).
    const onHistVis = () => {
      const visible = m.get("shared_histogram_visible");
      document.querySelectorAll(".ndv-hist-wrap").forEach((el) => {
        el.style.display = visible ? "" : "none";
      });
    };
    m.on("change:shared_histogram_visible", onHistVis);
    this._cleanups.push(() => m.off("change:shared_histogram_visible", onHistVis));
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    for (const cleanup of this._cleanups) cleanup();
    this._cleanups = [];
  }

  render() {
    return html`
      <div class="ndv-root">
        <div class="ndv-info-bar ${this._showDataInfo ? "" : "ndv-hidden"}">
          <span>${this._dataInfo}</span>
          <span>${this._hoverInfo}</span>
        </div>

        <ndv-dim-sliders
          class="${this._showControls ? "" : "ndv-hidden"}"
          .sliders=${this._sliders}
          .model=${this.model}
        ></ndv-dim-sliders>

        <ndv-lut-panel
          class="${this._showControls ? "" : "ndv-hidden"}"
          .luts=${this._luts}
          .model=${this.model}
          .showHistogramButton=${this._showHistogramButton}
          .useSharedHistogram=${this._useSharedHistogram}
        ></ndv-lut-panel>

        <ndv-toolbar
          class="${this._showControls ? "" : "ndv-hidden"}"
          .model=${this.model}
          .channelMode=${this._channelMode}
          .channelModeOptions=${this._channelModeOptions}
          .is3d=${this._is3d}
          .show3dButton=${this._show3dButton}
          .showChannelModeSelector=${this._showChannelModeSelector}
          .showResetZoomButton=${this._showResetZoomButton}
          .showRoiButton=${this._showRoiButton}
          .showHistogramButton=${this._showHistogramButton}
          .useSharedHistogram=${this._useSharedHistogram}
          .sharedHistogramVisible=${this._sharedHistogramVisible}
          .sharedHistogramLog=${this._sharedHistogramLog}
        ></ndv-toolbar>
      </div>
    `;
  }
}

if (!customElements.get("ndv-viewer")) customElements.define("ndv-viewer", NdvViewer);
