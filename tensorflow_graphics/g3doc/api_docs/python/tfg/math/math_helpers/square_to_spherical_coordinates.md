<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.math_helpers.square_to_spherical_coordinates" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.math_helpers.square_to_spherical_coordinates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/math_helpers.py">View source</a>



Maps points from a unit square to a unit sphere.

```python
tfg.math.math_helpers.square_to_spherical_coordinates(
    point_2d, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_2d`</b>: A tensor of shape `[A1, ..., An, 2]` with values in [0,1].
* <b>`name`</b>: A name for this op. Defaults to
  "math_square_to_spherical_coordinates".


#### Returns:

A tensor of shape `[A1, ..., An, 2]` with [..., 0] having values in
[0.0, pi] and [..., 1] with values in [0.0, 2pi].



#### Raises:


* <b>`ValueError`</b>: if the shape of `point_2d`  is not supported.
* <b>`InvalidArgumentError`</b>: if at least an element of `point_2d` is outside of
[0,1].