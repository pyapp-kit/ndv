from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qtpy.QtWidgets import QApplication

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLES_PY = list(EXAMPLES.glob("*.py"))


@pytest.fixture
def no_qapp_exec(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from qtpy.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "exec", lambda *_: None)
    yield
    # with suppress(Exception):
    #     if app := QApplication.instance():
    #         for wdg in QApplication.topLevelWidgets():
    #             wdg.close()
    #             wdg.deleteLater()
    #         app.processEvents()
    #         app.quit()


@pytest.mark.allow_leaks
@pytest.mark.usefixtures("no_qapp_exec")
@pytest.mark.parametrize("example", EXAMPLES_PY, ids=lambda x: x.name)
def test_example(qapp: QApplication, example: Path) -> None:
    runpy.run_path(str(example), run_name="__main__")
