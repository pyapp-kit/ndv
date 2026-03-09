# Embedding `ArrayViewer`

`ndv` can be embedded in an existing Qt (or wx) application and enriched with additional
elements in a custom layout. The following document shows some examples of such
implementation.

The key in each case is the use of the
[`ArrayViewer.widget`][ndv.controllers.ArrayViewer.widget] method, which returns
a native widget for the current GUI backend.

## Change the content of `ArrayViewer` via push buttons

The following script shows an example on how to dynamically select a data set
and load it in the `ArrayViewer`.

````python title="examples/cookbook/ndv_embedded.py"
--8<-- "examples/cookbook/ndv_embedded.py"
````

{{ screenshot: examples/cookbook/ndv_embedded.py }}

## Use multiple `ndv.ArrayViewer` controllers in the same widget

The following script shows an example on how to create multiple instances of the
`ArrayViewer` controller in the same widget and load two different datasets in
each one.

````python title="examples/cookbook/multi_ndv.py"
--8<-- "examples/cookbook/multi_ndv.py"
````

{{ screenshot: examples/cookbook/multi_ndv.py }}
