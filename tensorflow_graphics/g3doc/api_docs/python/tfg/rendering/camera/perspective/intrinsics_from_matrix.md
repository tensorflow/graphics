<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective.intrinsics_from_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.perspective.intrinsics_from_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



Extracts intrinsic parameters from a calibration matrix.

```python
tfg.rendering.camera.perspective.intrinsics_from_matrix(
    matrix, name=None
)
```



<!-- Placeholder for "Used in" -->

Extracts the focal length \\((f_x, f_y)\\) and the principal point
\\((c_x, c_y)\\) from a camera calibration matrix

$$
\mathbf{C} =
\begin{bmatrix}
f_x & 0 & c_x \\
0  & f_y & c_y \\
0  & 0  & 1 \\
\end{bmatrix}.
$$

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`matrix`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
  dimensions represent a camera calibration matrix.
* <b>`name`</b>: A name for this op that defaults to
  "perspective_intrinsics_from_matrix".


#### Returns:

Tuple of two tensors, each one of shape `[A1, ..., An, 2]`. The first
tensor represents the focal length, and the second one the principle point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `matrix` is not supported.