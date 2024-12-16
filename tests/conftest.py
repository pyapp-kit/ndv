from __future__ import annotations

import gc
import importlib
import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from ndv.views import gui_frontend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import FixtureRequest
    from qtpy.QtWidgets import QApplication


@pytest.fixture
def asyncio_app() -> Any:
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def any_app(request: pytest.FixtureRequest) -> Iterator[Any]:
    # this fixture will use the appropriate application depending on the env var
    # NDV_GUI_FRONTEND='qt' pytest
    # NDV_GUI_FRONTEND='jupyter' pytest

    if not importlib.util.find_spec("pytestqt"):
        # pytestqt isn't available ... this can't be a qt test
        os.environ["NDV_GUI_FRONTEND"] = "jupyter"

    if gui_frontend() == "qt":
        app = request.getfixturevalue("qapp")
        qtbot = request.getfixturevalue("qtbot")
        with patch.object(app, "exec", lambda *_: None):
            with _catch_qt_leaks(request, app):
                yield app, qtbot
    elif gui_frontend() == "jupyter":
        yield request.getfixturevalue("asyncio_app")


@contextmanager
def _catch_qt_leaks(request: FixtureRequest, qapp: QApplication) -> Iterator[None]:
    """Run after each test to ensure no widgets have been left around.

    When this test fails, it means that a widget being tested has an issue closing
    cleanly. Perhaps a strong reference has leaked somewhere.  Look for
    `functools.partial(self._method)` or `lambda: self._method` being used in that
    widget's code.
    """
    # check for the "allow_leaks" marker
    if "allow_leaks" in request.node.keywords:
        yield
        return

    nbefore = len(qapp.topLevelWidgets())
    failures_before = request.session.testsfailed
    yield
    # if the test failed, don't worry about checking widgets
    if request.session.testsfailed - failures_before:
        return
    try:
        from vispy.app.backends._qt import CanvasBackendDesktop

        allow: tuple[type, ...] = (CanvasBackendDesktop,)
    except ImportError:
        allow = ()

    # This is a known widget that is not cleaned up properly
    remaining = [w for w in qapp.topLevelWidgets() if not isinstance(w, allow)]
    if len(remaining) > nbefore:
        test_node = request.node

        test = f"{test_node.path.name}::{test_node.originalname}"
        msg = f"{len(remaining)} topLevelWidgets remaining after {test!r}:"

        for widget in remaining:
            try:
                obj_name = widget.objectName()
            except Exception:
                obj_name = None
            msg += f"\n{widget!r} {obj_name!r}"
            # Get the referrers of the widget
            referrers = gc.get_referrers(widget)
            msg += "\n  Referrers:"
            for ref in referrers:
                msg += f"\n  -   {ref}, {id(ref):#x}"

        raise AssertionError(msg)
