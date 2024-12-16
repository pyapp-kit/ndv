from __future__ import annotations

import runpy
from pathlib import Path

import pytest

try:
    import pytestqt  # noqa: F401
except ImportError:
    pytest.skip("This module requires qt frontend", allow_module_level=True)


EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_PY = list(EXAMPLES.glob("*.py"))


@pytest.mark.allow_leaks
@pytest.mark.usefixtures("any_app")
@pytest.mark.parametrize("example", EXAMPLES_PY, ids=lambda x: x.name)
def test_example(example: Path) -> None:
    try:
        runpy.run_path(str(example))
    except ImportError as e:
        pytest.skip(str(e))
