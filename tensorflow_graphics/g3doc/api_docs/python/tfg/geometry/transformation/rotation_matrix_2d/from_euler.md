<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_2d.from_euler" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_2d.from_euler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_2d.py">View source</a>



Converts an angle to a 2d rotation matrix.

```python
tfg.geometry.transformation.rotation_matrix_2d.from_euler(
    angle, name=None
)
```



<!-- Placeholder for "Used in" -->

Converts an angle $$\theta$$ to a 2d rotation matrix following the equation

$$
\mathbf{R} =
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}.
$$

#### Note:

The resulting matrix rotates points in the $$xy$$-plane counterclockwise.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`angle`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents an angle in radians.
* <b>`name`</b>: A name for this op that defaults to
  "rotation_matrix_2d_from_euler_angle".


#### Returns:

A tensor of shape `[A1, ..., An, 2, 2]`, where the last dimension represents
a 2d rotation matrix.



#### Raises:


* <b>`ValueError`</b>: If the shape of `angle` is not supported.