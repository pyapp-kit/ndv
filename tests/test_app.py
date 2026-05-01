from unittest.mock import patch

import ndv
import ndv.views._app


def test_set_gui_backend() -> None:
    backends = {
        "qt": ("ndv.views._qt._app", "QtAppWrap"),
        "jupyter": ("ndv.views._jupyter._app", "JupyterAppWrap"),
        "wx": ("ndv.views._wx._app", "WxAppWrap"),
    }
    with patch.object(ndv.views._app, "_load_app") as mock_load:
        for backend, import_tuple in backends.items():
            ndv.set_gui_backend(backend)
            ndv.views._app.ndv_app()
            mock_load.assert_called_once_with(*import_tuple)

            mock_load.reset_mock()
            ndv.views._app._APP = None
    ndv.set_gui_backend()


def test_set_canvas_backend() -> None:
    """Ensure that ndv.set_canvas_backend passes through to scenex."""
    backends = ["pygfx", "vispy"]
    for backend in backends:
        ndv.set_canvas_backend(backend)
        from scenex.adaptors._auto import determine_backend

        assert determine_backend() == backend
    ndv.set_canvas_backend()
