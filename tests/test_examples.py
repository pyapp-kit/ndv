from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

try:
    import pytestqt

    if pytestqt.qt_compat.qt_api.pytest_qt_api.startswith("pyside"):
        pytest.skip(
            "viewer still occasionally segfaults with pyside", allow_module_level=True
        )

except ImportError:
    pytest.skip("This module requires qt frontend", allow_module_level=True)


EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_PY = list(EXAMPLES.glob("*.py"))


# NOTE: I think a lot of the flakiness in this test is due to the fact that
# the examples may well recreate a QApplication, and that's problematic in a single
# process.  We would need to patch those calls a bit better to avoid that.
@pytest.mark.allow_leaks
@pytest.mark.usefixtures("any_app")
@pytest.mark.parametrize("example", EXAMPLES_PY, ids=lambda x: x.name)
@pytest.mark.filterwarnings("ignore:Downcasting integer data")
@pytest.mark.filterwarnings("ignore:.*Falling back to CPUScaledTexture")
@pytest.mark.skipif(
    bool(os.name == "nt" and os.getenv("CI")), reason="Test flaky on Windows CI"
)
def test_example(example: Path) -> None:
    try:
        runpy.run_path(str(example))
    except ImportError as e:
        pytest.skip(str(e))
