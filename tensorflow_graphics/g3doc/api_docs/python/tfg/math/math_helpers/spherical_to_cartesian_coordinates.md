<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.math_helpers.spherical_to_cartesian_coordinates" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.math_helpers.spherical_to_cartesian_coordinates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/math_helpers.py">View source</a>



Function to transform Cartesian coordinates to spherical coordinates.

```python
tfg.math.math_helpers.spherical_to_cartesian_coordinates(
    point_spherical, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_spherical`</b>: A tensor of shape `[A1, ..., An, 3]`. The last dimension
  contains r, theta, and phi that respectively correspond to the radius,
  polar angle and azimuthal angle; r must be non-negative.
* <b>`name`</b>: A name for this op. Defaults to 'spherical_to_cartesian_coordinates'.


#### Raises:


* <b>`tf.errors.InvalidArgumentError`</b>: If r, theta or phi contains out of range
data.


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension contains the
cartesian coordinates in x,y,z order.
