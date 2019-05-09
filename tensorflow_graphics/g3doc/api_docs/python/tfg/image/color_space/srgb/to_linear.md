<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.color_space.srgb.to_linear" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.color_space.srgb.to_linear

Converts sRGB colors to linear colors.

``` python
tfg.image.color_space.srgb.to_linear(
    srgb,
    gamma=2.4,
    name=None
)
```



Defined in [`image/color_space/srgb.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/color_space/srgb.py).

<!-- Placeholder for "Used in" -->

#### Note:

In the following, A1 to An are optional batch dimensions.


#### Args:

* <b>`srgb`</b>: A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension
  represents sRGB values.
* <b>`gamma`</b>: A float gamma value to use for the conversion.
* <b>`name`</b>: A name for this op that defaults to "srgb_to_linear".


#### Raises:

* <b>`ValueError`</b>: If `srgb` has rank < 1 or has its last dimension not equal to 3.


#### Returns:

A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension represents
RGB values in linear color space.