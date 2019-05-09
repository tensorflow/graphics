<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.pyramid.split" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.pyramid.split

Generates the different levels of the pyramid.

``` python
tfg.image.pyramid.split(
    image,
    num_levels,
    name=None
)
```



Defined in [`image/pyramid.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/pyramid.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`image`</b>: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
  the height of the image, `W` the width of the image, and `C` the number of
  channels of the image.
* <b>`num_levels`</b>: The number of levels to generate.
* <b>`name`</b>: A name for this op that defaults to "pyramid_split".


#### Returns:

A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
`H_i` and `W_i` are the height and width of the image for the level i.


#### Raises:

* <b>`ValueError`</b>: If the shape of `image` is not supported.