# Embed `ArrayViewer` in a `QWidget`

`ndv` can be embedded in an existing Qt application and enriched with additional elements in a custom layout.
The following document shows some examples of such implementation.

## Change the content of `ArrayViewer` via push buttons.

The following script shows an example on how to dynamically select a data set and load it in the `ArrayViewer`.

````python title="examples/cookbook/ndv_embedded.py"
--8<-- "examples/cookbook/ndv_embedded.py"
````

## Use multiple `ndv.ArrayViewer` controllers in the same widget

The following script shows an example on how to create multiple instances of the `ArrayViewer` controller in the same widget and load two different datasets in each one.

````python title="examples/cookbook/multi_ndv.py"
--8<-- "examples/cookbook/multi_ndv.py"
````

## A minimal microscope dashboard using `openwfs`

You can use `ndv` to take an external image source (i.e. a widefield camera) and show its content in real-time in a custom widget embedding `ArrayViewer`.
The script below uses [`openwfs`](https://github.com/IvoVellekoop/openwfs) to generate synthetic images of a sample and continuously update the view, and allows to move the field of view over the X and Y axis.

````python title="examples/cookbook/microscope_dashboard.py"
--8<-- "examples/cookbook/microscope_dashboard.py"
````