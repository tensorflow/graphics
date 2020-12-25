# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module implements perspective camera functionalities.

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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def parameters_from_right_handed(projection_matrix, name=None):
  """Recovers the parameters used to contruct a right handed projection matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    projection_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, containing
      matrices of right handed perspective-view frustum.
    name: A name for this op. Defaults to
      'perspective_parameters_from_right_handed'.

  Raises:
    InvalidArgumentError: if `projection_matrix` is not of the expected shape.

  Returns:
    Tuple of 4 tensors of shape `[A1, ..., An, 1]`, where the first tensor
    represents the vertical field of view used to contruct `projection_matrix,
    the second tensor represents the ascpect ratio used to construct
    `projection_matrix`, and the third and fourth parameters repectively
    represent the near and far clipping planes used to construct
    `projection_matrix`.
  """
  with tf.compat.v1.name_scope(name, "perspective_parameters_from_right_handed",
                               [projection_matrix]):
    projection_matrix = tf.convert_to_tensor(value=projection_matrix)

    shape.check_static(
        tensor=projection_matrix,
        tensor_name="projection_matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 4), (-1, 4)))

    inverse_tan_half_vertical_field_of_view = projection_matrix[..., 1, 1:2]
    vertical_field_of_view = 2.0 * tf.atan(
        1.0 / inverse_tan_half_vertical_field_of_view)
    aspect_ratio = inverse_tan_half_vertical_field_of_view / projection_matrix[
        ..., 0, 0:1]

    a = projection_matrix[..., 2, 2:3]
    b = projection_matrix[..., 2, 3:4]

    far = b / (a + 1.0)
    near = (a + 1.0) / (a - 1.0) * far

    return vertical_field_of_view, aspect_ratio, near, far


def right_handed(vertical_field_of_view, aspect_ratio, near, far, name=None):
  """Generates the matrix for a right handed perspective projection.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertical_field_of_view: A tensor of shape `[A1, ..., An, 1]`, where the last
      dimension represents the vertical field of view of the frustum expressed
      in radians. Note that values for `vertical_field_of_view` must be in the
      range (0,pi).
    aspect_ratio: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      stores the width over height ratio of the frustum. Note that values for
      `aspect_ratio` must be non-negative.
    near:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the near clipping plane. Note
      that values for `near` must be non-negative.
    far:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the far clipping plane. Note
      that values for `far` must be greater than those of `near`.
    name: A name for this op. Defaults to 'perspective_right_handed'.

  Raises:
    InvalidArgumentError: if any input contains data not in the specified range
      of valid values.
    ValueError: if the all the inputs are not of the same shape.

  Returns:
    A tensor of shape `[A1, ..., An, 4, 4]`, containing matrices of right
    handed perspective-view frustum.
  """
  with tf.compat.v1.name_scope(
      name, "perspective_right_handed",
      [vertical_field_of_view, aspect_ratio, near, far]):
    vertical_field_of_view = tf.convert_to_tensor(value=vertical_field_of_view)
    aspect_ratio = tf.convert_to_tensor(value=aspect_ratio)
    near = tf.convert_to_tensor(value=near)
    far = tf.convert_to_tensor(value=far)

    shape.check_static(
        tensor=vertical_field_of_view,
        tensor_name="vertical_field_of_view",
        has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=aspect_ratio, tensor_name="aspect_ratio", has_dim_equals=(-1, 1))
    shape.check_static(tensor=near, tensor_name="near", has_dim_equals=(-1, 1))
    shape.check_static(tensor=far, tensor_name="far", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(vertical_field_of_view, aspect_ratio, near, far),
        last_axes=-2,
        tensor_names=("vertical_field_of_view", "aspect_ratio", "near", "far"),
        broadcast_compatible=False)

    vertical_field_of_view = asserts.assert_all_in_range(
        vertical_field_of_view, 0.0, math.pi, open_bounds=True)
    aspect_ratio = asserts.assert_all_above(aspect_ratio, 0.0, open_bound=True)
    near = asserts.assert_all_above(near, 0.0, open_bound=True)
    far = asserts.assert_all_above(far, near, open_bound=True)

    inverse_tan_half_vertical_field_of_view = 1.0 / tf.tan(
        vertical_field_of_view * 0.5)
    zero = tf.zeros_like(inverse_tan_half_vertical_field_of_view)
    one = tf.ones_like(inverse_tan_half_vertical_field_of_view)
    near_minus_far = near - far
    matrix = tf.concat(
        (inverse_tan_half_vertical_field_of_view / aspect_ratio, zero, zero,
         zero, zero, inverse_tan_half_vertical_field_of_view, zero, zero, zero,
         zero, (far + near) / near_minus_far, 2.0 * far * near / near_minus_far,
         zero, zero, -one, zero),
        axis=-1)
    matrix_shape = tf.shape(input=matrix)
    output_shape = tf.concat((matrix_shape[:-1], (4, 4)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def intrinsics_from_matrix(matrix, name=None):
  r"""Extracts intrinsic parameters from a calibration matrix.

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

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a camera calibration matrix.
    name: A name for this op that defaults to
      "perspective_intrinsics_from_matrix".

  Returns:
    Tuple of two tensors, each one of shape `[A1, ..., An, 2]`. The first
    tensor represents the focal length, and the second one the principle point.

  Raises:
    ValueError: If the shape of `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_intrinsics_from_matrix",
                               [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3), (-2, 3)))

    fx = matrix[..., 0, 0]
    fy = matrix[..., 1, 1]
    cx = matrix[..., 0, 2]
    cy = matrix[..., 1, 2]
    focal = tf.stack((fx, fy), axis=-1)
    principal_point = tf.stack((cx, cy), axis=-1)
  return focal, principal_point


def matrix_from_intrinsics(focal, principal_point, name=None):
  r"""Builds calibration matrix from intrinsic parameters.

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

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    focal: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a camera focal length.
    principal_point: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension represents a camera principal point.
    name: A name for this op that defaults to
      "perspective_matrix_from_intrinsics".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a camera calibration matrix.

  Raises:
    ValueError: If the shape of `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_matrix_from_intrinsics",
                               [focal, principal_point]):
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)

    shape.check_static(
        tensor=focal, tensor_name="focal", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=principal_point,
        tensor_name="principal_point",
        has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(focal, principal_point),
        tensor_names=("focal", "principal_point"),
        last_axes=-2,
        broadcast_compatible=False)

    fx, fy = tf.unstack(focal, axis=-1)
    cx, cy = tf.unstack(principal_point, axis=-1)
    zero = tf.zeros_like(fx)
    one = tf.ones_like(fx)
    matrix = tf.stack((fx, zero, cx,
                       zero, fy, cy,
                       zero, zero, one),
                      axis=-1)  # pyformat: disable
    matrix_shape = tf.shape(input=matrix)
    output_shape = tf.concat((matrix_shape[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def project(point_3d, focal, principal_point, name=None):
  r"""Projects a 3d point onto the 2d camera plane.

  Projects a 3d point \\((x, y, z)\\) to a 2d point \\((x', y')\\) onto the
  image plane with

  $$
  \begin{matrix}
  x' = \frac{f_x}{z}x + c_x, & y' = \frac{f_y}{z}y + c_y,
  \end{matrix}
  $$

  where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
  point.

  Note:
    In the following, A1 to An are optional batch dimensions that must be
    broadcast compatible.

  Args:
    point_3d: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to project.
    focal: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a camera focal length.
    principal_point: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension represents a camera principal point.
    name: A name for this op that defaults to "perspective_project".

  Returns:
    A tensor of shape `[A1, ..., An, 2]`, where the last dimension represents
    a 2d point.

  Raises:
    ValueError: If the shape of `point_3d`, `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_project",
                               [point_3d, focal, principal_point]):
    point_3d = tf.convert_to_tensor(value=point_3d)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)

    shape.check_static(
        tensor=point_3d, tensor_name="point_3d", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=focal, tensor_name="focal", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=principal_point,
        tensor_name="principal_point",
        has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(point_3d, focal, principal_point),
        tensor_names=("point_3d", "focal", "principal_point"),
        last_axes=-2,
        broadcast_compatible=True)

    point_2d, depth = tf.split(point_3d, (2, 1), axis=-1)
    point_2d *= safe_ops.safe_signed_div(focal, depth)
    point_2d += principal_point
  return point_2d


def ray(point_2d, focal, principal_point, name=None):
  r"""Computes the 3d ray for a 2d point (the z component of the ray is 1).

  Computes the 3d ray \\((r_x, r_y, 1)\\) from the camera center to a 2d point
  \\((x', y')\\) on the image plane with

  $$
  \begin{matrix}
  r_x = \frac{(x' - c_x)}{f_x}, & r_y = \frac{(y' - c_y)}{f_y}, & z = 1,
  \end{matrix}
  $$

  where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
  point. The camera optical center is assumed to be at \\((0, 0, 0)\\).

  Note:
    In the following, A1 to An are optional batch dimensions that must be
    broadcast compatible.

  Args:
    point_2d: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point.
    focal: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a camera focal length.
    principal_point: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension represents a camera principal point.
    name: A name for this op that defaults to "perspective_ray".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d ray.

  Raises:
    ValueError: If the shape of `point_2d`, `focal`, or `principal_point` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_ray",
                               [point_2d, focal, principal_point]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)

    shape.check_static(
        tensor=point_2d, tensor_name="point_2d", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=focal, tensor_name="focal", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=principal_point,
        tensor_name="principal_point",
        has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(point_2d, focal, principal_point),
        tensor_names=("point_2d", "focal", "principal_point"),
        last_axes=-2,
        broadcast_compatible=True)

    point_2d -= principal_point
    point_2d = safe_ops.safe_signed_div(point_2d, focal)
    padding = [[0, 0] for _ in point_2d.shape]
    padding[-1][-1] = 1
    return tf.pad(
        tensor=point_2d, paddings=padding, mode="CONSTANT", constant_values=1.0)


def unproject(point_2d, depth, focal, principal_point, name=None):
  r"""Unprojects a 2d point in 3d.

  Unprojects a 2d point \\((x', y')\\) to a 3d point \\((x, y, z)\\) knowing the
  depth \\(z\\) with

  $$
  \begin{matrix}
  x = \frac{z (x' - c_x)}{f_x}, & y = \frac{z(y' - c_y)}{f_y}, & z = z,
  \end{matrix}
  $$

  where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
  point.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point_2d: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point to unproject.
    depth: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents the depth of a 2d point.
    focal: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a camera focal length.
    principal_point: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension represents a camera principal point.
    name: A name for this op that defaults to "perspective_unproject".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If the shape of `point_2d`, `depth`, `focal`, or
    `principal_point` is not supported.
  """
  with tf.compat.v1.name_scope(name, "perspective_unproject",
                               [point_2d, depth, focal, principal_point]):
    point_2d = tf.convert_to_tensor(value=point_2d)
    depth = tf.convert_to_tensor(value=depth)
    focal = tf.convert_to_tensor(value=focal)
    principal_point = tf.convert_to_tensor(value=principal_point)

    shape.check_static(
        tensor=point_2d, tensor_name="point_2d", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=depth, tensor_name="depth", has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=focal, tensor_name="focal", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=principal_point,
        tensor_name="principal_point",
        has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(point_2d, depth, focal, principal_point),
        tensor_names=("point_2d", "depth", "focal", "principal_point"),
        last_axes=-2,
        broadcast_compatible=False)

    point_2d -= principal_point
    point_2d *= safe_ops.safe_signed_div(depth, focal)
    return tf.concat((point_2d, depth), axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
