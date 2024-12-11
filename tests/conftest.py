from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from ndv.views import gui_frontend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import FixtureRequest
    from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def find_leaks(request: FixtureRequest, qapp: QApplication) -> Iterator[None]:
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
    from vispy.app.backends._qt import CanvasBackendDesktop

    # This is a known widget that is not cleaned up properly
    remaining = [
        w for w in qapp.topLevelWidgets() if not isinstance(w, CanvasBackendDesktop)
    ]
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
    if gui_frontend() == "qt":
        yield request.getfixturevalue("qapp")
    elif gui_frontend() == "jupyter":
        yield request.getfixturevalue("asyncio_app")
