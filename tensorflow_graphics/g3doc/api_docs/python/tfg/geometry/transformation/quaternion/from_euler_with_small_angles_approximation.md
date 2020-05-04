<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.from_euler_with_small_angles_approximation" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.from_euler_with_small_angles_approximation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Converts small Euler angles to quaternions.

```python
tfg.geometry.transformation.quaternion.from_euler_with_small_angles_approximation(
    angles, name=None
)
```



<!-- Placeholder for "Used in" -->

Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
approximated by their second order Taylor expansions, where
$$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
In the current implementation, the smallness of the angles is not verified.

#### Note:

Uses the z-y-x rotation convention (Tait-Bryan angles).



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`angles`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the three Euler angles. `[..., 0]` is the angle about `x` in
  radians, `[..., 1]` is the angle about `y` in radians and `[..., 2]` is the
  angle about `z` in radians.
 name: A name for this op that defaults to "quaternion_from_euler".


#### Returns:

A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
a normalized quaternion.



#### Raises:


* <b>`ValueError`</b>: If the shape of `angles` is not supported.