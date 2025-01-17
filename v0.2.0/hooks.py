import sys
from pathlib import Path
from typing import Any

import markdown

HERE = Path(__file__).parent


def on_startup(*_: Any, **__: Any) -> None:
    sys.path.append(str(HERE))


def on_config(config: dict, **__: Any) -> None:
    config["markdown_extensions"].append("hooks")


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


def makeExtension() -> "markdown.Extension":
    class MyExtension(markdown.Extension):
        def extendMarkdown(self, md: markdown.Markdown) -> None:
            # must be < 32 which is what pymdownx.snippets uses
            md.preprocessors.register(GhAdmonitionPreprocessor(), "snippet-post", 2)

    return MyExtension()
