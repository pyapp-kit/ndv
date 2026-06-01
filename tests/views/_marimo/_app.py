import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    from ndv.controllers._array_viewer import ArrayViewer

    arr = np.random.randint(0, 255, (3, 2, 64, 64), dtype=np.uint8)
    viewer = ArrayViewer(arr)
    viewer.display_model.channel_mode = "composite"
    viewer._view.frontend_widget()
    return


if __name__ == "__main__":
    app.run()
