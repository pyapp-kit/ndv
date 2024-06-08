import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest
    from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _find_leaks(request: "FixtureRequest", qapp: "QApplication") -> Iterator[None]:
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
    remaining = qapp.topLevelWidgets()
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
