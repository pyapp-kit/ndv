from __future__ import annotations

import re
import runpy
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, no_type_check
from unittest.mock import patch

import markdown
from mkdocs.plugins import get_plugin_logger
from mkdocs.structure.files import File, InclusionLevel
from qtpy.QtCore import QBuffer, QCoreApplication, QIODevice
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from matplotlib.dviread import Page
    from mkdocs.config import Config
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files


# logger for logging to mkdocs status
LOGGER = get_plugin_logger("ndv_docs")
# Path to the docs directory
DOCS = Path(__file__).parent
# pattern to detect {{ screenshot: some/file.py }}
SCREENSHOT_RE = re.compile(r"{{\s*screenshot:\s*(.+?)\s*}}")
# a mapping of {hash -> File} for all screenshots we've generated
SCREENSHOTS: dict[int, File] = {}


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool) -> None:
    sys.path.append(str(DOCS))


def on_config(config: Config, **__: Any) -> None:
    config["markdown_extensions"].append("hooks")


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    def get_screenshot_link(match: re.Match) -> str:
        script = Path(match.group(1))
        script_hash = hash(script.read_bytes())
        if not (file := SCREENSHOTS.get(script_hash)):
            LOGGER.info(f"Generating screenshot for {script}")
            SCREENSHOTS[script_hash] = file = File.generated(
                config,
                f"screenshots/{script.stem}.png",
                content=generate_screenshot(script),
                inclusion=InclusionLevel.NOT_IN_NAV,
            )

        files.append(file)
        x = f"![{script}](../{file.src_uri}){{ .auto-screenshot }}"
        return x

    new_markdown = SCREENSHOT_RE.sub(get_screenshot_link, markdown)
    return new_markdown


# ---------------------------- ScreenShot Generation ----------------------------


def generate_screenshot(script: Path) -> bytes:
    """Generate a screenshot from a script."""
    script = Path(script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"File not found: {script}")
    if script.suffix != ".py":
        raise ValueError(f"Invalid file type: {script}")

    original_new = QApplication.__new__

    # run
    @no_type_check
    def _start_app(*_: Any) -> QCoreApplication:
        if (app := QApplication.instance()) is None:
            QApplication._APP = app = original_new(*_)
            QApplication.__init__(app, [])
        return app

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)

    patch_app = patch.object(QApplication, "__new__", _start_app)
    patch_exec = patch.object(
        QApplication, "exec", partial(grab_top_widget, buffer=buffer)
    )
    with patch_app, patch_exec:
        runpy.run_path(str(script), run_name="__main__")

    return buffer.data().data()


def grab_top_widget(*_: Any, buffer: QBuffer, fmt: str = "png") -> None:
    """Mock QApplication.exec that snaps a screenshot and exits without blocking."""
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
        time.sleep(0.05)
        QApplication.processEvents()

    pixmap = main_wdg.grab()
    pixmap.save(buffer, fmt)

    main_wdg.close()
    main_wdg.deleteLater()

    for wdg in QApplication.topLevelWidgets():
        wdg.close()
        wdg.deleteLater()
    QApplication.processEvents()


# ---------------------------- Custom Markdown Extension ----------------------------


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
