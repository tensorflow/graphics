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
r"""This module implements 2d rotation matrix functionalities.

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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def from_euler(angle: type_alias.TensorLike,
               name: str = "rotation_matrix_2d_from_euler_angle") -> tf.Tensor:
  r"""Converts an angle to a 2d rotation matrix.

  Converts an angle $$\theta$$ to a 2d rotation matrix following the equation

  $$
  \mathbf{R} =
  \begin{bmatrix}
  \cos(\theta) & -\sin(\theta) \\
  \sin(\theta) & \cos(\theta)
  \end{bmatrix}.
  $$

  Note:
    The resulting matrix rotates points in the $$xy$$-plane counterclockwise.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle in radians.
    name: A name for this op that defaults to
      "rotation_matrix_2d_from_euler_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 2, 2]`, where the last dimension represents
    a 2d rotation matrix.

  Raises:
    ValueError: If the shape of `angle` is not supported.
  """
  with tf.name_scope(name):
    angle = tf.convert_to_tensor(value=angle)

    shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))

    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=angle)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def from_euler_with_small_angles_approximation(
    angles: type_alias.TensorLike,
    name: str = "rotation_matrix_2d_from_euler_with_small_angles_approximation"
) -> tf.Tensor:
  r"""Converts an angle to a 2d rotation matrix under the small angle assumption.

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

  Note:
    The resulting matrix rotates points in the $$xy$$-plane counterclockwise.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a small angle in radians.
    name: A name for this op that defaults to
      "rotation_matrix_2d_from_euler_with_small_angles_approximation".

  Returns:
    A tensor of shape `[A1, ..., An, 2, 2]`, where the last dimension represents
    a 2d rotation matrix.

  Raises:
    ValueError: If the shape of `angle` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 1))

    cos_angle = 1.0 - 0.5 * angles * angles
    sin_angle = angles
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=angles)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def inverse(matrix: type_alias.TensorLike,
            name: str = "rotation_matrix_2d_inverse") -> tf.Tensor:
  """Computes the inverse of a 2D rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 2, 2]`, where the last two
      dimensions represent a 2d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_2d_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 2, 2]`, where the last dimension represents
    a 2d rotation matrix.

  Raises:
    ValueError: If the shape of `matrix` is not supported.
  """
  with tf.name_scope(name):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 2), (-1, 2)))

    ndims = matrix.shape.ndims
    perm = list(range(ndims - 2)) + [ndims - 1, ndims - 2]
    return tf.transpose(a=matrix, perm=perm)


def is_valid(matrix: type_alias.TensorLike,
             atol: type_alias.Float = 1e-3,
             name: str = "rotation_matrix_2d_is_valid") -> tf.Tensor:
  r"""Determines if a matrix is a valid rotation matrix.

  Determines if a matrix $$\mathbf{R}$$ is a valid rotation matrix by checking
  that $$\mathbf{R}^T\mathbf{R} = \mathbf{I}$$ and $$\det(\mathbf{R}) = 1$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 2, 2]`, where the last two
      dimensions represent a 2d rotation matrix.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "rotation_matrix_2d_is_valid".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]` where False indicates
    that the input is not a valid rotation matrix.
  """
  with tf.name_scope(name):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 2), (-1, 2)))

    return rotation_matrix_common.is_valid(matrix, atol)


def rotate(point: type_alias.TensorLike,
           matrix: type_alias.TensorLike,
           name: str = "rotation_matrix_2d_rotate") -> tf.Tensor:
  """Rotates a 2d point using a 2d rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    identical.

  Args:
    point: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point.
    matrix: A tensor of shape `[A1, ..., An, 2, 2]`, where the last two
      dimensions represent a 2d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_2d_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point.

  Raises:
    ValueError: If the shape of `point` or `matrix` is not supported.
  """
  with tf.name_scope(name):
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 2), (-1, 2)))
    shape.compare_batch_dimensions(
        tensors=(point, matrix),
        tensor_names=("point", "matrix"),
        last_axes=(-2, -3),
        broadcast_compatible=True)

    point = tf.expand_dims(point, axis=-1)
    common_batch_shape = shape.get_broadcasted_shape(point.shape[:-2],
                                                     matrix.shape[:-2])

    def dim_value(dim: Optional[int] = None) -> int:
      return 1 if dim is None else tf.compat.dimension_value(dim)

    common_batch_shape = [dim_value(dim) for dim in common_batch_shape]
    point = tf.broadcast_to(point, common_batch_shape + [2, 1])
    matrix = tf.broadcast_to(matrix, common_batch_shape + [2, 2])
    rotated_point = tf.matmul(matrix, point)
    return tf.squeeze(rotated_point, axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
