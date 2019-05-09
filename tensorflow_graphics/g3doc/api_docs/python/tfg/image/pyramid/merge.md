<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.pyramid.merge" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.pyramid.merge

Merges the different levels of the pyramid back to an image.

``` python
tfg.image.pyramid.merge(
    levels,
    name=None
)
```



Defined in [`image/pyramid.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/pyramid.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`levels`</b>: A list containing tensors of shape `[B, H_i, W_i, C]`, where `B` is
  the batch size, H_i and W_i are the height and width of the image for the
  level i, and `C` the number of channels of the image.
* <b>`name`</b>: A name for this op that defaults to "pyramid_merge".


#### Returns:

A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
the height of the image, `W` the width of the image, and `C` the number of
channels of the image.


#### Raises:

* <b>`ValueError`</b>: If the shape of the elements of `levels` is not supported.