<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.from_euler_with_small_angles_approximation" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.from_euler_with_small_angles_approximation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



Converts small Euler angles to an axis-angle representation.

```python
tfg.geometry.transformation.axis_angle.from_euler_with_small_angles_approximation(
    angles, name=None
)
```



<!-- Placeholder for "Used in" -->

Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
approximated by their second order Taylor expansions, where
$$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
In the current implementation, the smallness of the angles is not verified.

#### Note:

The conversion is performed by first converting to a quaternion
representation, and then by converting the quaternion to an axis-angle.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`angles`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
  about `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians
  and `[A1, ..., An, 2]` is the angle about `z` in radians.
* <b>`name`</b>: A name for this op that defaults to
  "axis_angle_from_euler_with_small_angles_approximation".


#### Returns:

A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
`[A1, ..., An, 1]`, where the first tensor represents the axis, and the
second represents the angle. The resulting axis is a normalized vector.
