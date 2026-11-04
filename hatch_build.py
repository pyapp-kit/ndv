from __future__ import annotations

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook that runs the JS build before packaging."""

    def initialize(self, version, build_data):
        js_dir = Path(self.root) / "js"
        static_dir = Path(self.root) / "src" / "ndv" / "views" / "_jupyter" / "static"
        # Skip if already built (editable install re-runs)
        if (static_dir / "ndv-jupyter.js").exists():
            return
        subprocess.check_call(["npm", "ci"], cwd=js_dir)
        subprocess.check_call(["npm", "run", "build"], cwd=js_dir)
