import numpy as np
from rich import print
from vispy import app, io, scene

from ndv._chunking import iter_chunk_aligned_slices

vol1 = np.load(io.load_data_file("volume/stent.npz"))["arr_0"].astype(np.uint16)

canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
view = canvas.central_widget.add_view()
print("--------- create vol")
volume1 = scene.Volume(
    np.empty_like(vol1), parent=view.scene, texture_format="auto", clim=(0, 1200)
)
print("--------- create cam")
view.camera = scene.cameras.ArcballCamera(parent=view.scene, name="Arcball")

# Generate new data to update a subset of the volume


slices = iter_chunk_aligned_slices(
    vol1.shape, chunks=(32, 32, 32), slices=(slice(None), slice(None), slice(None))
)

for slice in list(slices)[::1]:
    offset = (x.start for x in slice)
    chunk = vol1[slice]
    # Update the texture with the new data at the calculated offset
    print("--------- update vol")
    volume1._texture._set_data(chunk, offset=tuple(offset))
canvas.update()
print("--------- run app")
app.run()
