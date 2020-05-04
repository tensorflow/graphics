<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_3d.from_euler_with_small_angles_approximation" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_3d.from_euler_with_small_angles_approximation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py">View source</a>



Convert an Euler angle representation to a rotation matrix.

```python
tfg.geometry.transformation.rotation_matrix_3d.from_euler_with_small_angles_approximation(
    angles, name=None
)
```



<!-- Placeholder for "Used in" -->

The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
approximated by their second order Taylor expansions, where
$$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
In the current implementation, the smallness of the angles is not verified.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`angles`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
  about `x` in radians, `[A1, ..., An, 1]` is the angle about `y` in radians
  and `[A1, ..., An, 2]` is the angle about `z` in radians.
* <b>`name`</b>: A name for this op that defaults to "rotation_matrix_3d_from_euler".


#### Returns:

A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
represent a 3d rotation matrix.



#### Raises:


* <b>`ValueError`</b>: If the shape of `angles` is not supported.