<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_2d.from_euler_with_small_angles_approximation" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_2d.from_euler_with_small_angles_approximation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_2d.py">View source</a>



Converts an angle to a 2d rotation matrix under the small angle assumption.

```python
tfg.geometry.transformation.rotation_matrix_2d.from_euler_with_small_angles_approximation(
    angles, name=None
)
```



<!-- Placeholder for "Used in" -->

Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
approximated by their second order Taylor expansions, where
$$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$. The 2d
rotation matrix will then be approximated as

$$
\mathbf{R} =
\begin{bmatrix}
1.0 - 0.5\theta^2 & -\theta \\
\theta & 1.0 - 0.5\theta^2
\end{bmatrix}.
$$

 In the current implementation, the smallness of the angles is not verified.

#### Note:

The resulting matrix rotates points in the $$xy$$-plane counterclockwise.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`angles`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents a small angle in radians.
* <b>`name`</b>: A name for this op that defaults to
  "rotation_matrix_2d_from_euler_with_small_angles_approximation".


#### Returns:

A tensor of shape `[A1, ..., An, 2, 2]`, where the last dimension represents
a 2d rotation matrix.



#### Raises:


* <b>`ValueError`</b>: If the shape of `angle` is not supported.