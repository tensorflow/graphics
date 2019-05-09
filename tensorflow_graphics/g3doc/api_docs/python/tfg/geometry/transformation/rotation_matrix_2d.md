<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_2d" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.geometry.transformation.rotation_matrix_2d

This module implements 2d rotation matrix functionalities.



Defined in [`geometry/transformation/rotation_matrix_2d.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_2d.py).

<!-- Placeholder for "Used in" -->

Given an angle of rotation $$\theta$$ a 2d rotation matrix can be expressed as

$$
\mathbf{R} =
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}.
$$

More details rotation matrices can be found on [this page.]
(https://en.wikipedia.org/wiki/Rotation_matrix)

Note: This matrix rotates points in the $$xy$$-plane counterclockwise.

## Functions

[`from_euler(...)`](../../../tfg/geometry/transformation/rotation_matrix_2d/from_euler.md): Converts an angle to a 2d rotation matrix.

[`from_euler_with_small_angles_approximation(...)`](../../../tfg/geometry/transformation/rotation_matrix_2d/from_euler_with_small_angles_approximation.md): Converts an angle to a 2d rotation matrix under the small angle assumption.

[`inverse(...)`](../../../tfg/geometry/transformation/rotation_matrix_2d/inverse.md): Computes the inverse of a 2D rotation matrix.

[`is_valid(...)`](../../../tfg/geometry/transformation/rotation_matrix_2d/is_valid.md): Determines if a matrix is a valid rotation matrix.

[`rotate(...)`](../../../tfg/geometry/transformation/rotation_matrix_2d/rotate.md): Rotates a 2d point using a 2d rotation matrix.

