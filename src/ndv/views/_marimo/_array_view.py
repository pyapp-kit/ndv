from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ndv.views._jupyter._array_view import JupyterArrayView

if TYPE_CHECKING:
    from ndv.models._viewer_model import ArrayViewerModel

_HIST_WRAP = (
    '<div class="ndv-hist-wrap" style="display:none;overflow:hidden">'
    "%(hist_html)s"
    "</div>"
)


class MarimoArrayView(JupyterArrayView):
    """ArrayView for marimo -- composes widgets via mo.vstack instead of VBox."""

    def __init__(
        self,
        canvas_widget: Any,
        viewer_model: ArrayViewerModel,
    ) -> None:
        super().__init__(canvas_widget, viewer_model)
        import marimo

        self._mo = marimo
        self._mo_canvas = marimo.ui.anywidget(self._canvas_widget)
        self._mo_histogram: Any | None = None

    def set_histogram_widget(self, widget: Any) -> None:
        super().set_histogram_widget(widget)
        self._mo_histogram = self._mo.ui.anywidget(widget)

    def frontend_widget(self) -> Any:
        parts: list[Any] = [self._mo_canvas, self._widget]
        if self._mo_histogram is not None:
            hist_html = self._mo.as_html(self._mo_histogram).text
            parts.append(self._mo.Html(_HIST_WRAP % {"hist_html": hist_html}))
        return self._mo.vstack(parts, gap=0)

    def _set_histogram_visible(self, visible: bool) -> None:
        # Visibility is toggled via DOM event from ndv-viewer JS.
        pass

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._mo.output.replace(self.frontend_widget())
