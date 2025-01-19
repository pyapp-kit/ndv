"""Tests pertaining to Qt components"""

from importlib.util import find_spec

import pytest

if not find_spec("ipywidgets"):
    pytest.skip(
        "Skipping Jupyter tests as Jupyter is not installed", allow_module_level=True
    )
