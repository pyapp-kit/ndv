from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ndv.views.bases._app import NDVApp

if TYPE_CHECKING:
    from ndv.views.bases import ArrayView


class JupyterAppWrap(NDVApp):
    """Provider for Jupyter notebooks/lab (NOT ipython)."""

    def is_running(self) -> bool:
        if ipy_shell := self._ipython_shell():
            return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
        return False

    def create_app(self) -> Any:
        if not self.is_running() and not os.getenv("PYTEST_CURRENT_TEST"):
            # if we got here, it probably means that someone used
            # NDV_GUI_FRONTEND=jupyter without actually being in a jupyter notebook
            # we allow it in tests, but not in normal usage.
            raise RuntimeError(  # pragma: no cover
                "Jupyter is not running a notebook shell.  Cannot create app."
            )

        # No app creation needed...
        # but make sure we can actually import the stuff we need
        import ipywidgets  # noqa: F401
        import jupyter  # noqa: F401

    def array_view_class(self) -> type[ArrayView]:
        from ._array_view import JupyterArrayView

        return JupyterArrayView
