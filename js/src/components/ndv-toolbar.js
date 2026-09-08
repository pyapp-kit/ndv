import { html, LitElement } from "lit";

export class NdvToolbar extends LitElement {
  createRenderRoot() {
    return this;
  }

  static properties = {
    model: { type: Object },
    channelMode: { type: String },
    channelModeOptions: { type: Array },
    is3d: { type: Boolean },
    show3dButton: { type: Boolean },
    showChannelModeSelector: { type: Boolean },
    showResetZoomButton: { type: Boolean },
    showRoiButton: { type: Boolean },
    showHistogramButton: { type: Boolean },
    useSharedHistogram: { type: Boolean },
    sharedHistogramVisible: { type: Boolean },
    sharedHistogramLog: { type: Boolean },
  };

  constructor() {
    super();
    this.model = null;
    this.channelMode = "grayscale";
    this.channelModeOptions = [];
    this.is3d = false;
    this.show3dButton = true;
    this.showChannelModeSelector = true;
    this.showResetZoomButton = true;
    this.showRoiButton = false;
    this.showHistogramButton = true;
    this.useSharedHistogram = false;
    this.sharedHistogramVisible = false;
    this.sharedHistogramLog = false;
  }

  _sendAction(type, extra = {}) {
    if (this.model) {
      this.model.set("_js_event", { type, ...extra });
      this.model.save_changes();
    }
  }

  _onChannelModeChange(e) {
    if (!this.model) return;
    this.model.set("channel_mode", e.target.value);
    this.model.save_changes();
  }

  _onNdimToggle() {
    this._sendAction("ndim_toggle", { value: !this.is3d });
  }

  _onResetZoom() {
    this._sendAction("reset_zoom");
  }

  _onRoiToggle() {
    this._sendAction("roi_toggle", { value: true });
  }

  _onSharedHistogramToggle() {
    if (!this.sharedHistogramVisible) {
      this._sendAction("shared_histogram_request");
    }
    if (this.model) {
      this.model.set("shared_histogram_visible", !this.sharedHistogramVisible);
      this.model.save_changes();
    }
  }

  _onSharedHistLogToggle() {
    if (this.model) {
      this.model.set("shared_histogram_log", !this.sharedHistogramLog);
      this.model.save_changes();
    }
  }

  render() {
    return html`
      <div class="ndv-toolbar">
        <wa-button
          size="small"
          appearance=${this.sharedHistogramVisible ? "filled" : "outlined"}
          class="${this.useSharedHistogram ? "" : "ndv-hidden"}"
          @click=${this._onSharedHistogramToggle}
          title="Toggle shared histogram"
        >
          <wa-icon name="chart-area" label="histogram"></wa-icon>
        </wa-button>

        <wa-button
          size="small"
          appearance=${this.sharedHistogramLog ? "filled" : "outlined"}
          class="${this.sharedHistogramVisible ? "" : "ndv-hidden"}"
          @click=${this._onSharedHistLogToggle}
          title="Log scale"
        >
          Log
        </wa-button>

        <div class="ndv-spacer"></div>

        <wa-select
          size="small"
          value=${this.channelMode}
          class="${this.showChannelModeSelector ? "" : "ndv-hidden"}"
          @change=${this._onChannelModeChange}
          hoist
        >
          ${(this.channelModeOptions || []).map(
            (opt) =>
              html`<wa-option value=${opt.value} ?disabled=${!opt.enabled}>
                ${opt.label}
              </wa-option>`,
          )}
        </wa-select>

        <wa-button
          size="small"
          appearance=${this.is3d ? "filled" : "outlined"}
          class="${this.show3dButton ? "" : "ndv-hidden"}"
          @click=${this._onNdimToggle}
          title="${this.is3d ? "Switch to 2D" : "Switch to 3D"}"
        >
          ${this.is3d ? "2D" : "3D"}
        </wa-button>

        <wa-button
          size="small"
          appearance="outlined"
          class="${this.showRoiButton ? "" : "ndv-hidden"}"
          @click=${this._onRoiToggle}
          title="Add ROI"
        >
          ROI
        </wa-button>

        <wa-button
          size="small"
          appearance="outlined"
          class="${this.showResetZoomButton ? "" : "ndv-hidden"}"
          @click=${this._onResetZoom}
          title="Reset zoom"
        >
          <wa-icon name="expand" label="reset zoom"></wa-icon>
        </wa-button>
      </div>
    `;
  }
}

if (!customElements.get("ndv-toolbar"))
  customElements.define("ndv-toolbar", NdvToolbar);
