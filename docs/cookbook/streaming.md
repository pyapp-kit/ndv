# Streaming updates

`ndv` can be used to visualize data that is continuously updated, such as
images from a camera or a live data stream. The following document shows some
examples of such implementation.

## Basic streaming, with no history

To visualize a live data stream, simply create an `ndv.ArrayViewer` controller
with an empty buffer matching your data shape.  Then, when new data is available,
update the buffer in place with the new data. Calling `update()` on the
[`ArrayDisplayModel.current_index`][ndv.models.ArrayDisplayModel]
will force the display to fetch your new data:

````python title="examples/streaming.py"
--8<-- "examples/streaming.py"
````

## Streaming, remembering the last N frames

To visualize a live data stream while keeping the last N frames in memory,
you can use the [`ndv.models.RingBuffer`][] class.  It offers a convenient
`append()` method to add new data, and takes care of updating the "apparent"
shape of the data (as far as the viewer is concerned):

````python title="examples/streaming_with_history.py"
--8<-- "examples/streaming_with_history.py"
````
