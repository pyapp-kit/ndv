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
from ndv.views._app import GUI_ENV_VAR, GuiFrontend

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop
    from collections.abc import Iterator

    import wx
    from pytest import FixtureRequest
    from qtpy.QtWidgets import QApplication


@pytest.fixture
def asyncio_app() -> Iterator[AbstractEventLoop]:
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def wxapp() -> Iterator[wx.App]:
    import wx

    if (_wxapp := wx.App.Get()) is None:
        _wxapp = wx.App()
    yield _wxapp


@pytest.fixture
def any_app(request: pytest.FixtureRequest) -> Iterator[Any]:
    # this fixture will use the appropriate application depending on the env var
    # NDV_GUI_FRONTEND='qt' pytest
    # NDV_GUI_FRONTEND='jupyter' pytest
    try:
        frontend = gui_frontend()
    except RuntimeError:
        # if we don't find any frontend, and jupyter is available, use that
        # since it requires very little setup
        if importlib.util.find_spec("jupyter"):
            os.environ[GUI_ENV_VAR] = "jupyter"
            gui_frontend.cache_clear()

        frontend = gui_frontend()

    if frontend == GuiFrontend.QT:
        app = request.getfixturevalue("qapp")
        qtbot = request.getfixturevalue("qtbot")
        with patch.object(app, "exec", lambda *_: app.processEvents()):
            with _catch_qt_leaks(request, app):
                yield app, qtbot
    elif frontend == GuiFrontend.JUPYTER:
        yield request.getfixturevalue("asyncio_app")
    elif frontend == GuiFrontend.WX:
        yield request.getfixturevalue("wxapp")
    else:
        raise RuntimeError("No GUI frontend found")


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
    except (ImportError, RuntimeError):
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
