# Contributing to ndv

Contributions are welcome. Please don't hesitate to open an issue if you have
any questions about the structure/design of the codebase.

## Setup

We recommend using `uv` when developing ndv. `uv` is an awesome tool that
manages virtual environments, python interpreters, and package dependencies.
Because `ndv` aims to support so many combinations of GUI and graphics
frameworks, it's not unusual to have multiple virtual environments for
different combinations of dependencies.  `uv` makes this easy.

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/),
then clone the repository:

```bash
git clone https://github.com/pyapp-kit/ndv
cd ndv
uv sync
```

This will create a virtual environment in `.venv` and install the "standard"
set of dev dependencies (pretty much everything except for PySide6).
You can then activate the environment with:

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```cmd
.\.venv\Scripts\activate
```

## Testing

As usual, you can run the tests with `pytest`:

```bash
uv run pytest
```

(Or just `pytest` if you've already activate your environment).

### Testing with different dependencies

To test different variants of `pygfx`, `vispy`, `pyqt`, `pyside`, `wx`:
use extras or groups to add specific members of
`project.optional-dependencies` or `project.dependency-groups`
declared in `pyproject.toml`.

```bash
# Run tests (default: pyqt + pygfx)
uv run --exact --no-dev --extra=pygfx --group=pyqt pytest

# Test with different backends
uv run --exact --no-dev --extra=vispy --group=jupyter pytest

# Test with array-libs group
uv run --exact --no-dev --extra=pygfx --group=pyqt --group=array-libs pytest
```

**Note:** These commands *will* recreate your current .venv folder,
since they include the `--exact` flag. If you don't want your current
env modified, add `--isolated` to the command.

```bash
uv run --exact --no-dev --isolated --extra=vispy --group=jupyter pytest
```

(Alternatively, just run `uv sync` again afterwards and it will bring
back the full env)

### Testing different Python versions

Use `UV_PYTHON` or `-p` to specify a different python version.

```bash
UV_PYTHON=3.10 uv run --exact --no-dev --extra=pygfx --group=pyqt pytest
```

### Testing with minimum dependency versions

To test against the minimum stated versions of dependencies, add
`--resolution=lowest-direct`:

```bash
uv run --exact --no-dev --extra=pygfx --group=pyqt \
  --resolution=lowest-direct pytest
```

## Linting and Formatting

To lint and format the code, use pre-commit (make sure you've run `uv sync` first):

```bash
uv run pre-commit run --all-files
```

## Building Documentation

To serve the documentation locally, use:

```bash
uv run --group docs mkdocs serve --no-strict
```

or to build into a `site` directory:

```bash
uv run --group docs mkdocs build --strict
```

If the screenshot generation is annoying, you can
disable it with the `GEN_SCREENSHOTS` environment variable:

```bash
GEN_SCREENSHOTS=0 uv run --group docs mkdocs build --strict
```

## Releasing

To release a new version, generate the changelog:

```sh
github_changelog_generator --future-release vX.Y.Z
```

then review it and commit it:

```sh
git commit -am "chore: changelog for vX.Y.Z"
```

then tag the commit and push:

```sh
git tag -a vX.Y.Z -m "vX.Y.Z"
git push upstream --follow-tags
```
