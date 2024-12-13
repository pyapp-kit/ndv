"""Example usage of new mvc pattern."""

from ndv import data, imshow

imshow(
    data.cells3d(),
    default_lut={"cmap": "cubehelix"},
    current_index={0: 20},
    visible_axes=(0, 2),
)
