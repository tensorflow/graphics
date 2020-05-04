<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective.matrix_from_intrinsics" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.perspective.matrix_from_intrinsics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



Builds calibration matrix from intrinsic parameters.

```python
tfg.rendering.camera.perspective.matrix_from_intrinsics(
    focal, principal_point, name=None
)
```



<!-- Placeholder for "Used in" -->

Builds the camera calibration matrix as

$$
\mathbf{C} =
\begin{bmatrix}
f_x & 0 & c_x \\
0  & f_y & c_y \\
0  & 0  & 1 \\
\end{bmatrix}
$$

from the focal length \\((f_x, f_y)\\) and the principal point
\\((c_x, c_y)\\).

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`focal`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a camera focal length.
* <b>`principal_point`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension represents a camera principal point.
* <b>`name`</b>: A name for this op that defaults to
  "perspective_matrix_from_intrinsics".


#### Returns:

A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
represent a camera calibration matrix.



#### Raises:


* <b>`ValueError`</b>: If the shape of `focal`, or `principal_point` is not
supported.