<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.from_rotation_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.from_rotation_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



Converts a rotation matrix to an axis-angle representation.

```python
tfg.geometry.transformation.axis_angle.from_rotation_matrix(
    rotation_matrix, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the current version the returned axis-angle representation is not unique
for a given rotation matrix. Since a direct conversion would not really be
faster, we first transform the rotation matrix to a quaternion, and finally
perform the conversion from that quaternion to the corresponding axis-angle
representation.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`rotation_matrix`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
  dimensions represent a rotation matrix.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_from_rotation_matrix".


#### Returns:

A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
`[A1, ..., An, 1]`, where the first tensor represents the axis, and the
second represents the angle. The resulting axis is a normalized vector.



#### Raises:


* <b>`ValueError`</b>: If the shape of `rotation_matrix` is not supported.