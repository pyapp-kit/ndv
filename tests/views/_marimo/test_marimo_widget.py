from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

pytest_plugins = ["pytest_playwright"]

import pytest  # noqa: E402

playwright = pytest.importorskip("playwright")

# Skip entire module if JS bundle hasn't been built (requires `cd js && npm run build`)
_STATIC = Path(__file__).parents[3] / "src" / "ndv" / "views" / "_jupyter" / "static"
if not (_STATIC / "ndv-jupyter.js").exists():
    pytest.skip("JS not built (run `cd js && npm run build`)", allow_module_level=True)

if TYPE_CHECKING:
    from playwright.sync_api import Page

TIMEOUT = 30_000  # ms


def test_widget_renders(marimo_url: str, page: Page) -> None:
    """Widget mounts and shows controls (sliders, LUT rows, toolbar)."""
    page.goto(marimo_url, wait_until="networkidle")

    # ndv-viewer custom element should appear
    viewer = page.locator("ndv-viewer")
    viewer.first.wait_for(state="attached", timeout=TIMEOUT)

    # Info bar with array shape
    info = page.locator(".ndv-info-bar")
    info.first.wait_for(state="visible", timeout=TIMEOUT)

    # At least one slider row
    slider = page.locator("ndv-dim-sliders .ndv-slider-row")
    slider.first.wait_for(state="visible", timeout=TIMEOUT)
    assert slider.count() >= 1

    # Toolbar should be present
    toolbar = page.locator("ndv-toolbar .ndv-toolbar")
    toolbar.first.wait_for(state="visible", timeout=TIMEOUT)

    # LUT rows exist (may be hidden during async channel resolution)
    lut_rows = page.locator("ndv-lut-row")
    lut_rows.first.wait_for(state="attached", timeout=TIMEOUT)
    assert lut_rows.count() >= 1


def test_slider_interaction(marimo_url: str, page: Page) -> None:
    """Moving a slider updates the value label."""
    page.goto(marimo_url, wait_until="networkidle")

    value_label = page.locator(".ndv-value-label").first
    value_label.wait_for(state="visible", timeout=TIMEOUT)
    initial_text = value_label.text_content()

    # Click near the right end of the slider track to change value
    slider_track = page.locator("ndv-dim-sliders wa-slider").first
    slider_track.wait_for(state="visible", timeout=TIMEOUT)
    box = slider_track.bounding_box()
    if box:
        page.mouse.click(box["x"] + box["width"] * 0.9, box["y"] + box["height"] / 2)
        page.wait_for_timeout(500)

    new_text = value_label.text_content()
    assert new_text != initial_text, "Slider value should change after click"


def test_histogram_toggle(marimo_url: str, page: Page) -> None:
    """Clicking the Hist button shows the histogram container."""
    page.goto(marimo_url, wait_until="networkidle")

    hist_wrap = page.locator(".ndv-hist-wrap")
    hist_wrap.first.wait_for(state="attached", timeout=TIMEOUT)

    # Should start hidden
    assert hist_wrap.first.is_hidden()

    # Click the Hist button
    hist_btn = page.locator("ndv-toolbar wa-button", has_text="Hist")
    hist_btn.first.wait_for(state="visible", timeout=TIMEOUT)
    hist_btn.first.click()
    page.wait_for_timeout(1000)

    # Should now be visible
    assert hist_wrap.first.is_visible()

    # Click again to hide
    hist_btn.first.click()
    page.wait_for_timeout(500)
    assert hist_wrap.first.is_hidden()
