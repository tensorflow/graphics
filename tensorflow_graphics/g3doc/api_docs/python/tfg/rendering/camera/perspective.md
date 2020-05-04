<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.rendering.camera.perspective

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



This module implements perspective camera functionalities.


The perspective camera model, also referred to as pinhole camera model, is
defined using a focal length \\((f_x, f_y)\\) and a principal point
\\((c_x, c_y)\\). The perspective camera model can be written as a calibration
matrix

$$
\mathbf{C} =
\begin{bmatrix}
f_x & 0 & c_x \\
0  & f_y & c_y \\
0  & 0  & 1 \\
\end{bmatrix},
$$

also referred to as the intrinsic parameter matrix. The camera focal length
\\((f_x, f_y)\\), defined in pixels, is the physical focal length divided by the
physical size of a camera pixel. The physical focal length is the distance
between the camera center and the image plane. The principal point is the
intersection of the camera axis with the image plane. The camera axis is the
line perpendicular to the image plane starting at the optical center.

More details about perspective cameras can be found on [this page.]
(http://ksimek.github.io/2013/08/13/intrinsic/)

Note: The current implementation does not take into account distortion or
skew parameters.

## Functions

[`intrinsics_from_matrix(...)`](../../../tfg/rendering/camera/perspective/intrinsics_from_matrix.md): Extracts intrinsic parameters from a calibration matrix.

[`matrix_from_intrinsics(...)`](../../../tfg/rendering/camera/perspective/matrix_from_intrinsics.md): Builds calibration matrix from intrinsic parameters.

[`project(...)`](../../../tfg/rendering/camera/perspective/project.md): Projects a 3d point onto the 2d camera plane.

[`ray(...)`](../../../tfg/rendering/camera/perspective/ray.md): Computes the 3d ray for a 2d point (the z component of the ray is 1).

[`unproject(...)`](../../../tfg/rendering/camera/perspective/unproject.md): Unprojects a 2d point in 3d.

