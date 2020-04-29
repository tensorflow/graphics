<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.from_quaternion" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.from_quaternion

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



Converts a quaternion to an axis-angle representation.

```python
tfg.geometry.transformation.axis_angle.from_quaternion(
    quaternion, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`quaternion`</b>: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a normalized quaternion.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_from_quaternion".


#### Returns:

Tuple of two tensors of shape `[A1, ..., An, 3]` and `[A1, ..., An, 1]`,
where the first tensor represents the axis, and the second represents the
angle. The resulting axis is a normalized vector.



#### Raises:


* <b>`ValueError`</b>: If the shape of `quaternion` is not supported.