from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"marimo did not start on port {port}")


@pytest.fixture(scope="module")
def marimo_url() -> Iterator[str]:
    """Start a headless marimo server and yield its URL."""
    port = _free_port()
    notebook = str(Path(__file__).parent / "_app.py")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "marimo",
            "run",
            "--headless",
            "--no-token",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            notebook,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # TODO: also test with vispy backend
        env={**__import__("os").environ, "NDV_CANVAS_BACKEND": "pygfx"},
    )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
