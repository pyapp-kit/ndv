"""Tests pertaining to Pygfx components"""

from importlib.util import find_spec

import pytest

if not find_spec("pygfx"):
    pytest.skip(
        "Skipping Pygfx tests as Pygfx is not installed", allow_module_level=True
    )
