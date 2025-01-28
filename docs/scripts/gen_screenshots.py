import io
import runpy
import sys
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, cast, overload
from unittest.mock import patch

from qtpy.QtCore import QBuffer, QIODevice
from qtpy.QtWidgets import QApplication


@overload
def generate_screenshot(script: Path, output: BinaryIO) -> BinaryIO: ...
@overload
def generate_screenshot(script: Path, output: Path | None = ...) -> Path: ...
def generate_screenshot(
    script: Path, output: Path | BinaryIO | None = None
) -> Path | BinaryIO:
    """Generate a screenshot from a script."""
    script = Path(script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"File not found: {script}")
    if script.suffix != ".py":
        raise ValueError(f"Invalid file type: {script}")

    if output is None:
        output = script.with_suffix(".png")
    if isinstance(output, Path):
        output.parent.mkdir(parents=True, exist_ok=True)
    elif not isinstance(output, io.BufferedWriter):
        raise TypeError(
            f"output must be a Path or file-like object, not {type(output)}"
        )

    # run
    with patch.object(QApplication, "exec", partial(snap_viewer, output=output)):
        runpy.run_path(str(script), run_name="__main__")

    return output


def snap_viewer(*_: Any, output: Path | BinaryIO) -> None:
    """Mock QApplication.exec that snaps a screenshot and exits without blocking."""
    from ndv.views._qt._array_view import _QArrayViewer

    try:
        qviewer = next(
            wdg
            for wdg in QApplication.topLevelWidgets()
            if isinstance(wdg, _QArrayViewer)
        )
    except StopIteration:
        return

    target: str | QIODevice
    if isinstance(output, io.BufferedWriter):
        target = QBuffer()
        target.open(QIODevice.OpenModeFlag.WriteOnly)
    else:
        target = str(output)

    QApplication.sendPostedEvents()
    QApplication.processEvents()

    qviewer.grab().save(target)
    if isinstance(target, QBuffer):
        img_bytes = target.data().data()
        cast(BinaryIO, output).write(img_bytes)

    for wdg in QApplication.topLevelWidgets():
        wdg.close()
        wdg.deleteLater()
    QApplication.processEvents()


# In this block, we're likely being run by mkdocs_gen_files
# to generate screenshots for the documentation.
# See extra.screenshots in mkdocs.yml for configuration to include/exclude
if __name__ == "<run_path>":
    import mkdocs_gen_files
    from mkdocs.plugins import get_plugin_logger

    logger = get_plugin_logger("gen_screenshots.py")
    FE = mkdocs_gen_files.editor.FilesEditor.current()
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    screenshots_dir = Path("screenshots")  # virtual directory inside of docs
    cfg: dict = FE.config.get("extra", {}).get("screenshots", {})  # type: ignore
    suffix = cfg.get("suffix", ".png")
    scripts = sorted(examples_dir.rglob("*.py"))
    if includes := cfg.get("include_patterns", []):
        scripts = [ex for ex in scripts if any(ex.match(inc) for inc in includes)]
        if includes and not scripts:
            logger.warning(
                "No scripts found with the patterns in "
                f"extra.screenshots.include_patterns: {includes!r}",
            )
            sys.exit(0)

    if excludes := cfg.get("exclude_patterns", []):
        scripts = [ex for ex in scripts if not any(ex.match(exc) for exc in excludes)]

    for script in scripts:
        rel_path = script.relative_to(examples_dir)
        script_screenshot_path = screenshots_dir / rel_path.with_suffix(suffix)
        with FE.open(str(script_screenshot_path), "wb") as fd:
            try:
                generate_screenshot(script, output=fd)
            except Exception as e:
                get_plugin_logger("gen_screenshots.py").warning(
                    f"Error generating screenshot for {script}: {e}"
                )
            else:
                relpath = script.relative_to(examples_dir.parent)
                logger.info(f"Generated screenshot for {relpath} to {fd.name}")


elif __name__ == "__main__":
    script = Path(__file__).parent.parent.parent / "examples" / "numpy_arr.py"
    screenshot = generate_screenshot(script)
