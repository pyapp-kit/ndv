# Examples

To quickly run any of the examples in this directory, without needing to install
either ndv or even python, you can [install
uv](https://docs.astral.sh/uv/getting-started/installation/), and then use [`uv
run`](https://docs.astral.sh/uv/guides/scripts/):

```shell
# locally within the ndv repo
uv run examples/numpy_arr.py

# from anywhere
uv run https://github.com/pyapp-kit/ndv/raw/refs/heads/main/examples/numpy_arr.py
```

Replace `numpy_arr.py` with the name of the example file in this directory you
want to run.

## Notebooks

To run any of the `.ipynb` files in this directory, you can use uv with
[`juv`](https://github.com/manzt/juv)

```shell
uvx juv run examples/notebook.ipynb
```

*At the time of writing, juv only appears to support local files, so you should
clone this repo first.*
