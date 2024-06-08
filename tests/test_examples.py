from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_PY = list(EXAMPLES.glob("*.py"))


@pytest.fixture
def no_qapp_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QApplication, "exec", lambda *_: None)


@pytest.mark.allow_leaks
@pytest.mark.usefixtures("no_qapp_exec")
@pytest.mark.parametrize("example", EXAMPLES_PY, ids=lambda x: x.name)
def test_example(qapp: QApplication, example: Path) -> None:
    try:
        runpy.run_path(str(example))
    except ImportError as e:
        pytest.skip(str(e))
