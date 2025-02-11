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

The makefile also has a few targets for running tests (these all
depend on having `uv` installed):

```bash
# just test with something standard (currently pyqt6/pygfx)
make test
```

### Testing with different dependencies

To test different variants of `pygfx`, `vispy`, `pyqt`, `pyside`, `wx`:  
use extras or groups to add specific members of
`project.optional-dependencies` or `project.dependency-groups`
declared in `pyproject.toml`.

```bash
# run all 
make test extras=pyqt,vispy groups=array-libs
```

> [!TIP]
> that above command actually has an alias:
>
> ```bash
> make test-arrays
> ```

**Note:** These commands *will* recreate your current .venv folder,
since they include the `--exact` flag. If you don't want your current
env modified, add `isolated=1` to the command.

```bash
make test extras=jupyter,vispy isolated=1
```

(Alternatively, just run `uv sync` again afterwards and it will bring
back the full env)

### Testing different Python versions

Use `py=` to specify a different python version.

```bash
make test py=3.10
```

### Testing with minimum dependency versions

To test against the minimum stated versions of dependencies, use `min=1`

```bash
make test min=1
```

## Linting and Formatting

To lint and format the code, use pre-commit (make sure you've run `uv sync` first):

```bash
uv run pre-commit run --all-files
```

or

```bash
make lint
```

## Building Documentation

To serve the documentation locally, use:

```bash
make docs-serve
```

or to build into a `site` directory:

```bash
make docs
```

If the screenshot generation is annoying, you can
disable it with the `GEN_SCREENSHOTS` environment variable:

```bash
GEN_SCREENSHOTS=0 make docs
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
