from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ndv.views._jupyter._app import JupyterAppWrap

if TYPE_CHECKING:
    from ndv.views.bases import ArrayView


class MarimoAppWrap(JupyterAppWrap):
    """Provider for marimo notebooks."""

    def is_running(self) -> bool:
        from ndv.views._app import _is_marimo_running

        return _is_marimo_running()

    def create_app(self) -> Any:
        if not self.is_running():
            raise RuntimeError("marimo is not running. Cannot create app.")

        import anywidget  # noqa: F401

    def array_view_class(self) -> type[ArrayView]:
        from ndv.views._marimo._array_view import MarimoArrayView

        return MarimoArrayView
