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
"""This module implements TensorFlow 3d rotation matrix utility functions.

More details rotation matrices can be found on [this page.]
(https://en.wikipedia.org/wiki/Rotation_matrix)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import tfg_flags

FLAGS = flags.FLAGS


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
  """Builds a rotation matrix from sines and cosines of Euler angles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  """
  sin_angles.shape.assert_is_compatible_with(cos_angles.shape)

  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  m00 = cy * cz
  m01 = (sx * sy * cz) - (cx * sz)
  m02 = (cx * sy * cz) + (sx * sz)
  m10 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m12 = (cx * sy * sz) - (sx * cz)
  m20 = -sy
  m21 = sx * cy
  m22 = cx * cy
  matrix = tf.stack((m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22),
                    axis=-1)  # pyformat: disable
  output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)


def assert_rotation_matrix_normalized(matrix, eps=1e-3, name=None):
  """Checks whether a matrix is a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a 3d rotation matrix.
    eps: The absolute tolerance parameter.
    name: A name for this op that defaults to
      'assert_rotation_matrix_normalized'.

  Returns:
    The input matrix, with dependence on the assertion operator in the graph.

  Raises:
    tf.errors.InvalidArgumentError: If rotation_matrix_3d is not normalized.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return matrix

  with tf.compat.v1.name_scope(name, "assert_rotation_matrix_normalized",
                               [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))

    is_matrix_normalized = is_valid(matrix, atol=eps)
    with tf.control_dependencies([
        tf.compat.v1.assert_equal(
            is_matrix_normalized,
            tf.ones_like(is_matrix_normalized, dtype=tf.bool))
    ]):
      return tf.identity(matrix)


def from_axis_angle(axis, angle, name=None):
  """Convert an axis-angle representation to a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a normalized axis.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_axis_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represents a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_axis_angle",
                               [axis, angle]):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)

    shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(axis, angle),
        tensor_names=("axis", "angle"),
        last_axes=-2,
        broadcast_compatible=True)
    axis = asserts.assert_normalized(axis)

    sin_axis = tf.sin(angle) * axis
    cos_angle = tf.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
    matrix = tf.stack((diag_x, m01, m02,
                       m10, diag_y, m12,
                       m20, m21, diag_z),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def from_euler(angles, name=None):
  r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_euler", [angles]):
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))

    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_euler_with_small_angles_approximation(angles, name=None):
  r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
      about `x` in radians, `[A1, ..., An, 1]` is the angle about `y` in radians
      and `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(
      name, "rotation_matrix_3d_from_euler_with_small_angles", [angles]):
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))

    sin_angles = angles
    cos_angles = 1.0 - 0.5 * tf.square(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_quaternion(quaternion, name=None):
  """Convert a quaternion to a rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_quaternion".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_quaternion",
                               [quaternion]):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    quaternion = asserts.assert_normalized(quaternion)

    x, y, z, w = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def inverse(matrix, name=None):
  """Computes the inverse of a 3D rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a 3d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_3d_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `matrix` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_inverse", [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))
    matrix = assert_rotation_matrix_normalized(matrix)

    ndims = matrix.shape.ndims
    perm = list(range(ndims - 2)) + [ndims - 1, ndims - 2]
    return tf.transpose(a=matrix, perm=perm)


def is_valid(matrix, atol=1e-3, name=None):
  """Determines if a matrix is a valid rotation matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last two
      dimensions represent a matrix.
    atol: Absolute tolerance parameter.
    name: A name for this op that defaults to "rotation_matrix_3d_is_valid".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]` where False indicates
    that the input is not a valid rotation matrix.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_is_valid", [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))

    return rotation_matrix_common.is_valid(matrix, atol)


def rotate(point, matrix, name=None):
  """Rotate a point using a rotation matrix 3d.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last dimension
      represents a 3d rotation matrix.
    name: A name for this op that defaults to "rotation_matrix_3d_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If the shape of `point` or `rotation_matrix_3d` is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_rotate",
                               [point, matrix]):
    point = tf.convert_to_tensor(value=point)
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))
    shape.compare_batch_dimensions(
        tensors=(point, matrix),
        tensor_names=("point", "matrix"),
        last_axes=(-2, -3),
        broadcast_compatible=True)
    matrix = assert_rotation_matrix_normalized(matrix)

    point = tf.expand_dims(point, axis=-1)
    common_batch_shape = shape.get_broadcasted_shape(
        point.shape[:-2], matrix.shape[:-2])
    def dim_value(dim):
      return 1 if dim is None else tf.compat.v1.dimension_value(dim)
    common_batch_shape = [dim_value(dim) for dim in common_batch_shape]
    point = tf.broadcast_to(point, common_batch_shape + [3, 1])
    matrix = tf.broadcast_to(matrix, common_batch_shape + [3, 3])
    rotated_point = tf.matmul(matrix, point)
    return tf.squeeze(rotated_point, axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
