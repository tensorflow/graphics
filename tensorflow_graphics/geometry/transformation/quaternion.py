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
"""This module implements TensorFlow quaternion utility functions.

A quaternion is written as $$q =  xi + yj + zk + w$$, where $$i,j,k$$ forms the
three bases of the imaginary part. The functions implemented in this file
use the Hamilton convention where $$i^2 = j^2 = k^2 = ijk = -1$$. A quaternion
is stored in a 4-D vector $$[x, y, z, w]^T$$.

More details about Hamiltonian quaternions can be found on [this page.]
(https://en.wikipedia.org/wiki/Quaternion)

Note: Some of the functions expect normalized quaternions as inputs where
$$x^2 + y^2 + z^2 + w^2 = 1$$.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def _build_quaternion_from_sines_and_cosines(
    sin_half_angles: type_alias.TensorLike,
    cos_half_angles: type_alias.TensorLike) -> tf.Tensor:
  """Builds a quaternion from sines and cosines of half Euler angles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sin_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the sine of half Euler angles.
    cos_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the cosine of half Euler angles.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a quaternion.
  """
  c1, c2, c3 = tf.unstack(cos_half_angles, axis=-1)
  s1, s2, s3 = tf.unstack(sin_half_angles, axis=-1)
  w = c1 * c2 * c3 + s1 * s2 * s3
  x = -c1 * s2 * s3 + s1 * c2 * c3
  y = c1 * s2 * c3 + s1 * c2 * s3
  z = -s1 * s2 * c3 + c1 * c2 * s3
  return tf.stack((x, y, z, w), axis=-1)


def between_two_vectors_3d(vector1: type_alias.TensorLike,
                           vector2: type_alias.TensorLike,
                           name: str = "quaternion_between_two_vectors_3d"
                           ) -> tf.Tensor:
  """Computes quaternion over the shortest arc between two vectors.

  Result quaternion describes shortest geodesic rotation from
  vector1 to vector2.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vector.
    vector2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vector.
    name: A name for this op that defaults to
      "quaternion_between_two_vectors_3d".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `vector1` or `vector2` is not supported.
  """
  with tf.name_scope(name):
    vector1 = tf.convert_to_tensor(value=vector1)
    vector2 = tf.convert_to_tensor(value=vector2)

    shape.check_static(
        tensor=vector1, tensor_name="vector1", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=vector2, tensor_name="vector2", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(vector1, vector2), last_axes=-2, broadcast_compatible=True)

    # Make sure that we are dealing with unit vectors.
    vector1 = tf.nn.l2_normalize(vector1, axis=-1)
    vector2 = tf.nn.l2_normalize(vector2, axis=-1)
    cos_theta = vector.dot(vector1, vector2)
    real_part = 1.0 + cos_theta
    axis = vector.cross(vector1, vector2)

    # Compute arbitrary antiparallel axes to rotate around in case of opposite
    # vectors.
    x, y, z = tf.split(vector1, (1, 1, 1), axis=-1)
    x_bigger_z = tf.abs(x) > tf.abs(z)
    x_bigger_z = tf.concat([x_bigger_z] * 3, axis=-1)
    antiparallel_axis = tf.where(x_bigger_z,
                                 tf.concat((-y, x, tf.zeros_like(z)), axis=-1),
                                 tf.concat((tf.zeros_like(x), -z, y), axis=-1))

    # Compute rotation between two vectors.
    is_antiparallel = real_part < 1e-6
    is_antiparallel = tf.concat([is_antiparallel] * 4, axis=-1)
    rot = tf.where(
        is_antiparallel,
        tf.concat((antiparallel_axis, tf.zeros_like(real_part)), axis=-1),
        tf.concat((axis, real_part), axis=-1))
    return tf.nn.l2_normalize(rot, axis=-1)


def conjugate(quaternion: type_alias.TensorLike,
              name: str = "quaternion_conjugate") -> tf.Tensor:
  """Computes the conjugate of a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_conjugate".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.name_scope(name):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))

    xyz, w = tf.split(quaternion, (3, 1), axis=-1)
    return tf.concat((-xyz, w), axis=-1)


def from_axis_angle(axis: type_alias.TensorLike,
                    angle: type_alias.TensorLike,
                    name: str = "quaternion_from_axis_angle"
                    ) -> tf.Tensor:
  """Converts an axis-angle representation to a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "quaternion_from_axis_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
  with tf.name_scope(name):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)

    shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(axis, angle), last_axes=-2, broadcast_compatible=True)
    axis = asserts.assert_normalized(axis)

    half_angle = 0.5 * angle
    w = tf.cos(half_angle)
    xyz = tf.sin(half_angle) * axis
    return tf.concat((xyz, w), axis=-1)


def from_euler(angles: type_alias.TensorLike,
               name: str = "quaternion_from_euler"
               ) -> tf.Tensor:
  """Converts an Euler angle representation to a quaternion.

  Note:
    Uses the z-y-x rotation convention (Tait-Bryan angles).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[..., 0]` is the angle about `x` in
      radians, `[..., 1]` is the angle about `y` in radians and `[..., 2]` is
      the angle about `z` in radians.
    name: A name for this op that defaults to "quaternion_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))

    half_angles = angles / 2.0
    cos_half_angles = tf.cos(half_angles)
    sin_half_angles = tf.sin(half_angles)
    return _build_quaternion_from_sines_and_cosines(sin_half_angles,
                                                    cos_half_angles)


def from_euler_with_small_angles_approximation(
    angles: type_alias.TensorLike,
    name: str = "quaternion_from_euler") -> tf.Tensor:
  r"""Converts small Euler angles to quaternions.

  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    Uses the z-y-x rotation convention (Tait-Bryan angles).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
   angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
     represents the three Euler angles. `[..., 0]` is the angle about `x` in
     radians, `[..., 1]` is the angle about `y` in radians and `[..., 2]` is the
     angle about `z` in radians.
    name: A name for this op that defaults to "quaternion_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(
        tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))

    half_angles = angles / 2.0
    cos_half_angles = 1.0 - 0.5 * half_angles * half_angles
    sin_half_angles = half_angles
    quaternion = _build_quaternion_from_sines_and_cosines(
        sin_half_angles, cos_half_angles)
    # We need to normalize the quaternion due to the small angle approximation.
    return tf.nn.l2_normalize(quaternion, axis=-1)


def from_rotation_matrix(rotation_matrix: type_alias.TensorLike,
                         name: str = "quaternion_from_rotation_matrix"
                         ) -> tf.Tensor:
  """Converts a rotation matrix representation to a quaternion.

  Warning:
    This function is not smooth everywhere.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "quaternion_from_rotation_matrix".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
  """
  with tf.name_scope(name):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

    shape.check_static(
        tensor=rotation_matrix,
        tensor_name="rotation_matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3), (-2, 3)))
    rotation_matrix = rotation_matrix_3d.assert_rotation_matrix_normalized(
        rotation_matrix)

    trace = tf.linalg.trace(rotation_matrix)
    eps_addition = asserts.select_eps_for_addition(rotation_matrix.dtype)
    rows = tf.unstack(rotation_matrix, axis=-2)
    entries = [tf.unstack(row, axis=-1) for row in rows]

    def tr_positive():
      sq = tf.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
      qw = 0.25 * sq
      qx = safe_ops.safe_unsigned_div(entries[2][1] - entries[1][2], sq)
      qy = safe_ops.safe_unsigned_div(entries[0][2] - entries[2][0], sq)
      qz = safe_ops.safe_unsigned_div(entries[1][0] - entries[0][1], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_1():
      sq = tf.sqrt(1.0 + entries[0][0] - entries[1][1] - entries[2][2] +
                   eps_addition) * 2.  # sq = 4 * qx.
      qw = safe_ops.safe_unsigned_div(entries[2][1] - entries[1][2], sq)
      qx = 0.25 * sq
      qy = safe_ops.safe_unsigned_div(entries[0][1] + entries[1][0], sq)
      qz = safe_ops.safe_unsigned_div(entries[0][2] + entries[2][0], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_2():
      sq = tf.sqrt(1.0 + entries[1][1] - entries[0][0] - entries[2][2] +
                   eps_addition) * 2.  # sq = 4 * qy.
      qw = safe_ops.safe_unsigned_div(entries[0][2] - entries[2][0], sq)
      qx = safe_ops.safe_unsigned_div(entries[0][1] + entries[1][0], sq)
      qy = 0.25 * sq
      qz = safe_ops.safe_unsigned_div(entries[1][2] + entries[2][1], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_3():
      sq = tf.sqrt(1.0 + entries[2][2] - entries[0][0] - entries[1][1] +
                   eps_addition) * 2.  # sq = 4 * qz.
      qw = safe_ops.safe_unsigned_div(entries[1][0] - entries[0][1], sq)
      qx = safe_ops.safe_unsigned_div(entries[0][2] + entries[2][0], sq)
      qy = safe_ops.safe_unsigned_div(entries[1][2] + entries[2][1], sq)
      qz = 0.25 * sq
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_idx(cond):
      cond = tf.expand_dims(cond, -1)
      cond = tf.tile(cond, [1] * (rotation_matrix.shape.ndims - 2) + [4])
      return cond

    where_2 = tf.where(
        cond_idx(entries[1][1] > entries[2][2]), cond_2(), cond_3())
    where_1 = tf.where(
        cond_idx((entries[0][0] > entries[1][1])
                 & (entries[0][0] > entries[2][2])), cond_1(), where_2)
    quat = tf.where(cond_idx(trace > 0), tr_positive(), where_1)
    return quat


def inverse(quaternion: type_alias.TensorLike,
            name: str = "quaternion_inverse"
            ) -> tf.Tensor:
  """Computes the inverse of a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.name_scope(name):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    quaternion = asserts.assert_normalized(quaternion)

    squared_norm = tf.reduce_sum(
        input_tensor=tf.square(quaternion), axis=-1, keepdims=True)
    return safe_ops.safe_unsigned_div(conjugate(quaternion), squared_norm)


def is_normalized(quaternion: type_alias.TensorLike,
                  atol: type_alias.Float = 1e-3,
                  name: str = "quaternion_is_normalized"
                  ) -> tf.Tensor:
  """Determines if quaternion is normalized quaternion or not.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "quaternion_is_normalized".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]`, where False indicates
    that the quaternion is not normalized.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.name_scope(name):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))

    norms = tf.norm(tensor=quaternion, axis=-1, keepdims=True)
    return tf.where(
        tf.abs(norms - 1.) < atol, tf.ones_like(norms, dtype=bool),
        tf.zeros_like(norms, dtype=bool))


def normalize(quaternion: type_alias.TensorLike,
              eps: type_alias.Float = 1e-12,
              name: str = "quaternion_normalize"
              ) -> tf.Tensor:
  """Normalizes a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    eps: A lower bound value for the norm that defaults to 1e-12.
    name: A name for this op that defaults to "quaternion_normalize".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where the quaternion elements have
    been normalized.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.name_scope(name):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))

    return tf.math.l2_normalize(quaternion, axis=-1, epsilon=eps)


def multiply(quaternion1: type_alias.TensorLike,
             quaternion2: type_alias.TensorLike,
             name: str = "quaternion_multiply"
             ) -> tf.Tensor:
  """Multiplies two quaternions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion1:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    quaternion2:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    name: A name for this op that defaults to "quaternion_multiply".

  Returns:
    A tensor of shape `[A1, ..., An, 4]` representing quaternions.

  Raises:
    ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
  """
  with tf.name_scope(name):
    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)

    shape.check_static(
        tensor=quaternion1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    shape.check_static(
        tensor=quaternion2, tensor_name="quaternion2", has_dim_equals=(-1, 4))

    x1, y1, z1, w1 = tf.unstack(quaternion1, axis=-1)
    x2, y2, z2, w2 = tf.unstack(quaternion2, axis=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return tf.stack((x, y, z, w), axis=-1)


def normalized_random_uniform(quaternion_shape: List[int],
                              name: str = "quaternion_normalized_random_uniform"
                              ) -> tf.Tensor:
  """Random normalized quaternion following a uniform distribution law on SO(3).

  Args:
    quaternion_shape: A list representing the shape of the output tensor.
    name: A name for this op that defaults to
      "quaternion_normalized_random_uniform".

  Returns:
    A tensor of shape `[quaternion_shape[0],...,quaternion_shape[-1], 4]`
    representing random normalized quaternions.
  """
  with tf.name_scope(name):
    quaternion_shape = tf.convert_to_tensor(
        value=quaternion_shape, dtype=tf.int32)
    quaternion_shape = tf.concat((quaternion_shape, tf.constant([4])), axis=0)
    random_normal = tf.random.normal(quaternion_shape)
  return normalize(random_normal)


def normalized_random_uniform_initializer():
  """Random unit quaternion initializer."""

  # Since variable initializers must take `shape` as input, we cannot prevent
  # a clash between util.shape and the argument here. Therefore we have to
  # disable redefined-outer-name for this function.
  # pylint: disable=redefined-outer-name
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    """Generate a random normalized quaternion.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      shape: A list representing the shape of the output. The last entry of the
        list must be `4`.
      dtype: type of the output (tf.float32 is the only type supported).
      partition_info: how the variable is partitioned (not used).

    Returns:
      A tensor of shape `[A1, ..., An, 4]` representing normalized quaternions.

    Raises:
      ValueError: If `shape` or `dtype` are not supported.
    """
    del partition_info  # unused
    if dtype != tf.float32:
      raise ValueError("'dtype' must be tf.float32.")
    if shape[-1] != 4:
      raise ValueError("Last dimension of 'shape' must be 4.")

    return normalized_random_uniform(shape[:-1])

  return _initializer
  # pylint: enable=redefined-outer-name


def rotate(point: type_alias.TensorLike,
           quaternion: type_alias.TensorLike,
           name: str = "quaternion_rotate"
           ) -> tf.Tensor:
  """Rotates a point using a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents a
    3d point.

  Raises:
    ValueError: If the shape of `point` or `quaternion` is not supported.
  """
  with tf.name_scope(name):
    point = tf.convert_to_tensor(value=point)
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    shape.compare_batch_dimensions(
        tensors=(point, quaternion), last_axes=-2, broadcast_compatible=True)
    quaternion = asserts.assert_normalized(quaternion)

    padding = [[0, 0] for _ in range(point.shape.ndims)]
    padding[-1][-1] = 1
    point = tf.pad(tensor=point, paddings=padding, mode="CONSTANT")
    point = multiply(quaternion, point)
    point = multiply(point, conjugate(quaternion))
    xyz, _ = tf.split(point, (3, 1), axis=-1)
    return xyz


def relative_angle(quaternion1: type_alias.TensorLike,
                   quaternion2: type_alias.TensorLike,
                   name: str = "quaternion_relative_angle"
                   ) -> tf.Tensor:
  r"""Computes the unsigned relative rotation angle between 2 unit quaternions.

  Given two normalized quanternions $$\mathbf{q}_1$$ and $$\mathbf{q}_2$$, the
  relative angle is computed as
  $$\theta = 2\arccos(\mathbf{q}_1^T\mathbf{q}_2)$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion1: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    quaternion2: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_relative_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 1]` where the last dimension represents
    rotation angles in the range [0.0, pi].

  Raises:
    ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
  """
  with tf.name_scope(name):
    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)

    shape.check_static(
        tensor=quaternion1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    shape.check_static(
        tensor=quaternion2, tensor_name="quaternion2", has_dim_equals=(-1, 4))
    quaternion1 = asserts.assert_normalized(quaternion1)
    quaternion2 = asserts.assert_normalized(quaternion2)

    dot_product = vector.dot(quaternion1, quaternion2, keepdims=False)
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 4.0 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(
        dot_product, -1.0, 1.0, False, eps=eps_dot_prod)
    return 2.0 * tf.acos(tf.abs(dot_product))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
