<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.color_space.srgb.from_linear_rgb" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.color_space.srgb.from_linear_rgb

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/color_space/srgb.py">View source</a>



Converts linear RGB to sRGB colors.

```python
tfg.image.color_space.srgb.from_linear_rgb(
    linear_rgb, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`linear_rgb`</b>: A Tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension
  represents RGB values in the range [0, 1] in linear color space.
* <b>`name`</b>: A name for this op that defaults to "srgb_from_linear_rgb".


#### Raises:


* <b>`ValueError`</b>: If `linear_rgb` has rank < 1 or has its last dimension not
  equal to 3.


#### Returns:

A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension represents
sRGB values.
