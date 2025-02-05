"""Tests pertaining to Qt components"""

from importlib.util import find_spec

import pytest

if not find_spec("wx"):
    pytest.skip("Skipping wx tests as wx is not installed", allow_module_level=True)
