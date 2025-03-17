from __future__ import annotations

from pytest import fixture

from ndv.views._qt._app import QtAppWrap


@fixture(autouse=True)
def init_provider() -> None:
    provider = QtAppWrap()
    provider.create_app()
