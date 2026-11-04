import { html, LitElement } from "lit";
import { repeat } from "lit/directives/repeat.js";

export class NdvDimSliders extends LitElement {
  createRenderRoot() {
    return this;
  }

  static properties = {
    sliders: { type: Array },
    model: { type: Object },
  };

  constructor() {
    super();
    this.sliders = [];
    this.model = null;
    this._dragging = new Set();
    // Play state per axis: { axis: { intervalId, startTime, startFrame, fps } }
    this._playing = new Map();
  }

  _onSliderInput(e, slider) {
    const value = e.target.value;
    this._dragging.add(slider.axis);
    if (this.model) {
      this.model.set("_js_event", {
        type: "slider_changed",
        axis: slider.axis,
        value,
      });
      this.model.save_changes();
    }
  }

  _onSliderChange(_e, slider) {
    this._dragging.delete(slider.axis);
  }

  _togglePlay(slider) {
    if (this._playing.has(slider.axis)) {
      this._stopPlay(slider.axis);
    } else {
      this._startPlay(slider);
    }
    this.requestUpdate();
  }

  _startPlay(slider, fps = 20) {
    const state = this._playing.get(slider.axis);
    if (state) clearInterval(state.intervalId);

    const startTime = performance.now();
    const startFrame = slider.value;
    const nFrames = slider.max - slider.min + 1;
    const interval = Math.max(1, Math.floor(1000 / fps / 2)); // oversample 2x

    const intervalId = setInterval(() => {
      const elapsed = (performance.now() - startTime) / 1000;
      const target =
        slider.min + ((startFrame - slider.min + Math.floor(elapsed * fps)) % nFrames);
      if (this.model) {
        this.model.set("_js_event", {
          type: "slider_changed",
          axis: slider.axis,
          value: target,
        });
        this.model.save_changes();
      }
    }, interval);

    this._playing.set(slider.axis, { intervalId, fps });
  }

  _stopPlay(axis) {
    const state = this._playing.get(axis);
    if (state) {
      clearInterval(state.intervalId);
      this._playing.delete(axis);
    }
  }

  _onFpsChange(e, slider) {
    const fps = parseFloat(e.target.value) || 20;
    if (this._playing.has(slider.axis)) {
      this._startPlay(slider, fps);
    }
    this.requestUpdate();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    for (const [axis] of this._playing) {
      this._stopPlay(axis);
    }
  }

  render() {
    return html`
      ${repeat(
        this.sliders,
        (s) => s.axis,
        (s) => {
          const playing = this._playing.has(s.axis);
          return html`
            <div class="ndv-slider-row ${s.visible ? "" : "ndv-hidden"}">
              <wa-button
                size="small"
                appearance=${playing ? "filled" : "plain"}
                class="ndv-play-btn"
                @click=${() => this._togglePlay(s)}
                title="Play/Pause"
              >
                <wa-icon name=${playing ? "pause" : "play"} label=${playing ? "Pause" : "Play"}></wa-icon>
              </wa-button>
              <span class="ndv-axis-label">${s.label}</span>
              <wa-slider
                size="small"
                min=${s.min}
                max=${s.max}
                .value=${this._dragging.has(s.axis) ? undefined : s.value}
                step=${s.step}
                with-tooltip
                @input=${(e) => this._onSliderInput(e, s)}
                @change=${(e) => this._onSliderChange(e, s)}
              ></wa-slider>
              <span class="ndv-value-label">${s.value} / ${s.max}</span>
            </div>
          `;
        },
      )}
    `;
  }
}

if (!customElements.get("ndv-dim-sliders"))
  customElements.define("ndv-dim-sliders", NdvDimSliders);
