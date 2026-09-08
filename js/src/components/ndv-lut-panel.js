import { html, LitElement } from "lit";
import { repeat } from "lit/directives/repeat.js";
import "./ndv-lut-row.js";

export class NdvLutPanel extends LitElement {
  createRenderRoot() {
    return this;
  }

  static properties = {
    luts: { type: Array },
    model: { type: Object },
    showHistogramButton: { type: Boolean },
    useSharedHistogram: { type: Boolean },
  };

  constructor() {
    super();
    this.luts = [];
    this.model = null;
    this.showHistogramButton = true;
    this.useSharedHistogram = false;
  }

  render() {
    return html`
      ${repeat(
        this.luts,
        (lut) => lut.key,
        (lut) => html`
          <ndv-lut-row
            .lut=${lut}
            .model=${this.model}
            .showHistogramButton=${this.showHistogramButton && !this.useSharedHistogram}
            style="${lut.row_visible ? "" : "display:none"}"
          ></ndv-lut-row>
        `,
      )}
    `;
  }
}

if (!customElements.get("ndv-lut-panel"))
  customElements.define("ndv-lut-panel", NdvLutPanel);
