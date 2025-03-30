"""Custom hooks used in our documentation.

This is registered in the `hooks` section of `mkdocs.yml` file and is used to apply
all of our customizations to the documentation build process, such as automatically
generating screenshots from scripts.

Screenshots can be inserted into the documentation by using the following syntax:

```markdown
{{ screenshot: some/file/relative/to/repo_root.py }}
```
"""

from __future__ import annotations

import os
import re
import runpy
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import markdown
from mkdocs.plugins import get_plugin_logger
from mkdocs.structure.files import File, InclusionLevel
from qtpy.QtCore import QBuffer, QCoreApplication, QIODevice
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from matplotlib.dviread import Page
    from mkdocs.config import Config
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files


# logger for logging to mkdocs status
LOGGER = get_plugin_logger("ndv_docs")
# Path to the docs directory
DOCS = Path(__file__).parent
ROOT = DOCS.parent
# pattern to detect {{ screenshot: some/file.py }}
SCREENSHOT_RE = re.compile(r"{{\s*screenshot:\s*(.+?)\s*}}")
# a mapping of {hash -> File} for all screenshots we've generated
SCREENSHOTS: defaultdict[int, set[File]] = defaultdict(set)
GEN_SCREENSHOTS = os.getenv("GEN_SCREENSHOTS", "1") not in ("0", "false", "False")


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool) -> None:
    """Runs once at the very beginning of an mkdocs invocation."""
    sys.path.append(str(DOCS))


def on_config(config: Config, **__: Any) -> None:
    """First event called on build, run immediately after user cfg is loaded/validated.

    Any alterations to the config should be made here.
    """
    config["markdown_extensions"].append("hooks")


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Called after the page's markdown is loaded from file.

    Can be used to alter the Markdown source text. The meta- data has been stripped off
    and is available as page.meta at this point.
    """
    # ------------------------- Screenshot Generation ---------------------------------
    #

    def get_screenshot_link(match: re.Match) -> str:
        """Callback for SCREENSHOT_RE.sub.

        Return a markdown link to replace `{{ screenshot: some/file.py }}`
        """
        script_path = Path(match.group(1))
        # add the script to the watch list so that mkdocs will rebuild the page
        # if the script is edited
        abs_path = str(ROOT / script_path)
        if abs_path not in config.watch:
            config.watch.append(abs_path)

        # if we've never seen the content of this script before, generate a screenshot
        # and store it in the SCREENSHOTS dict
        if not (ss_files := SCREENSHOTS[hash(script_path.read_bytes())]):
            LOGGER.info(f"Generating screenshot for {script_path}")
            for content, mode in generate_screenshots(script_path):
                src_uri = f"screenshots/{script_path.stem}_{mode}.png"
                LOGGER.info(f"   >> {mode}: {src_uri}")

                # mkdocs takes care of copying the file to the site_dir
                # we just have to create a File object for it
                file = File.generated(
                    config,
                    src_uri=src_uri,
                    content=content,
                    inclusion=InclusionLevel.NOT_IN_NAV,
                )
                ss_files.add(file)

        link = ""
        for file in ss_files:
            # build markdown links, with lazy loading and theme awareness
            files.append(file)
            mode = "dark" if "dark" in file.src_uri else "light"
            alt_txt = f"Screenshot generated with {script_path.stem}"
            uri = f"../{file.src_uri}#only-{mode}"
            # note, the .auto-screenshot class is styled in ndv.css
            link += f"![{alt_txt}]({uri}){{ .auto-screenshot loading=lazy }}\n"
        return link

    # Find all {{ screenshot: some/file.py }},
    # generate a screenshot for each file and replace the tag with the image link
    # this generates two links: one for light mode and one for dark mode
    if GEN_SCREENSHOTS:
        markdown = SCREENSHOT_RE.sub(get_screenshot_link, markdown)

    # ---------------------------------------------------------------------------

    return markdown


# ---------------------------- ScreenShot Generation ----------------------------

Mode = Literal["light", "dark"]


def generate_screenshots(script: Path) -> Iterable[tuple[bytes, Mode]]:
    """Generate a screenshot from a script."""
    script = Path(script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"File not found: {script}")
    if script.suffix != ".py":
        raise ValueError(f"Invalid file type: {script}")

    # patch any calls to QApplication([]) in scripts, so they don't cause an
    # error if the Application is already created.
    original_new = QApplication.__new__

    def _start_app(*_: Any) -> QCoreApplication:
        if (app := QApplication.instance()) is None:
            QApplication._APP = app = original_new(*_)  # type: ignore
            QApplication.__init__(app, [])  # type: ignore
        return app

    patch_app = patch.object(QApplication, "__new__", _start_app)

    # We'll collect yields from grab_top_widget into a list,
    # then yield them after the script has finished.
    results: list[tuple[bytes, Mode]] = []

    # patch QApplication.exec to grab screenshots instead of running the event loop

    def patched_exec(_: Any) -> int:
        # Call the generator and store items in results
        for item in grab_top_widget():
            results.append(item)
        # Return 0, or whatever exit code you like, so that
        # QApp thinks exec() ended
        return 0

    patch_exec = patch.object(QApplication, "exec", patched_exec)
    with patch_app, patch_exec:
        # run the script
        runpy.run_path(str(script), run_name="__main__")

    # return the screenshot grabbed by the grab_top_widget function
    return results


def grab_top_widget(
    fmt: str = "png",
) -> Iterator[tuple[bytes, Literal["light", "dark"]]]:
    """Find the top widget and return bytes containing a screenshot."""
    from qtpy.QtWidgets import QFrame

    try:
        main_wdg = next(
            wdg for wdg in QApplication.topLevelWidgets() if not isinstance(wdg, QFrame)
        )
    except StopIteration:
        return

    QApplication.sendPostedEvents()
    # the sleep here is mostly to allow the widget to render,
    # the biggest issue at the time is the animated expansion of the luts
    # QcollapsibleWidget.  That should be fixed at superqt to not animate if desired.
    for _i in range(10):
        time.sleep(0.1)
        QApplication.processEvents()

    def _get_bytes() -> bytes:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap = main_wdg.grab()
        pixmap.save(buffer, fmt)
        return buffer.data().data()

    modes: list[Mode] = ["dark", "light"] if sys.platform == "darwin" else ["light"]
    for mode in modes:
        set_dark_mode(mode == "dark")
        for _i in range(12):
            time.sleep(0.1)
            QApplication.processEvents()
        yield _get_bytes(), mode

    for wdg in QApplication.topLevelWidgets():
        wdg.close()
        wdg.deleteLater()
    QApplication.processEvents()
    QApplication.processEvents()


def set_dark_mode(bool: bool) -> bool:
    if not sys.platform == "darwin":
        return False
    try:
        subprocess.check_call(
            [
                "osascript",
                "-e",
                'tell application "System Events" to tell appearance preferences '
                f"to set dark mode to {bool}",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------- Custom Markdown Extension ----------------------------

# this stuff can be removed if we stop using snippets to insert markdown
# (like README.md) that has github style admonitions.


class GhAdmonitionPreprocessor(markdown.preprocessors.Preprocessor):
    """Convert github-style admonitions to MkDocs-style admonitions."""

    def convert_gh_admonitions(self, lines: list[str]) -> list[str]:
        """Transforms [!X] blocks in Markdown into !!! X blocks."""
        output_lines = []
        inside_note_block = False
        type_ = None
        note_content = []

        for line in lines:
            if line.startswith("> [!"):  # Start of a note block
                inside_note_block = True
                type_ = line.split("[!", 1)[1].split("]", 1)[0].lower()
            elif inside_note_block and line.startswith(
                "> "
            ):  # Content of the note block
                note_content.append(line[2:])  # Strip the '> ' prefix
            else:
                if inside_note_block:  # End of the note block
                    output_lines.append(f"!!! {type_}")
                    output_lines.extend(
                        f"    {content_line}" for content_line in note_content
                    )
                    inside_note_block = False
                    type_ = None
                    note_content = []

                output_lines.append(line)  # Add non-note lines as is

        # If the file ends with a note block, finalize it
        if inside_note_block:
            output_lines.append(f"!!! {type_}")
            output_lines.extend(f"    {content_line}" for content_line in note_content)

        return output_lines

    def run(self, lines: list[str]) -> list[str]:
        return self.convert_gh_admonitions(lines)


def makeExtension() -> markdown.Extension:
    class MyExtension(markdown.Extension):
        def extendMarkdown(self, md: markdown.Markdown) -> None:
            # must be < 32 which is what pymdownx.snippets uses
            md.preprocessors.register(GhAdmonitionPreprocessor(), "snippet-post", 2)

    return MyExtension()
