import marimo as mo

__generated_with = "0.22.0"
app = mo.App()


@app.cell
def _():
    from ndv import data
    from ndv.controllers._array_viewer import ArrayViewer

    viewer = ArrayViewer(data.cells3d())
    viewer.display_model.channel_mode = "composite"
    viewer.display_model.current_index.update({0: 32})
    viewer._view.frontend_widget()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
