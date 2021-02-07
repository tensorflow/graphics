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
r"""This module implements axis-angle functionalities.

The axis-angle representation is defined as $$\theta\mathbf{a}$$, where
$$\mathbf{a}$$ is a unit vector indicating the direction of rotation and
$$\theta$$ is a scalar controlling the angle of rotation. It is important to
note that the axis-angle does not perform rotation by itself, but that it can be
used to rotate any given vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into
a vector $$\mathbf{v}'$$ using the Rodrigues' rotation formula:

$$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
+\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

More details about the axis-angle formalism can be found on [this page.]
(https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)

Note: Some of the functions defined in the module expect
a normalized axis $$\mathbf{a} = [x, y, z]^T$$ as inputs where
$$x^2 + y^2 + z^2 = 1$$.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import quaternion as quaternion_lib
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def from_euler(angles, name="axis_angle_from_euler"):
  r"""Converts Euler angles to an axis-angle representation.

  Note:
    The conversion is performed by first converting to a quaternion
    representation, and then by converting the quaternion to an axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "axis_angle_from_euler".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.
  """
  with tf.name_scope(name):
    quaternion = quaternion_lib.from_euler(angles)
    return from_quaternion(quaternion)


def from_euler_with_small_angles_approximation(
    angles, name="axis_angle_from_euler_with_small_angles_approximation"):
  r"""Converts small Euler angles to an axis-angle representation.

  Under the small angle assumption, $$\sin(x)$$ and $$\cos(x)$$ can be
  approximated by their second order Taylor expansions, where
  $$\sin(x) \approx x$$ and $$\cos(x) \approx 1 - \frac{x^2}{2}$$.
  In the current implementation, the smallness of the angles is not verified.

  Note:
    The conversion is performed by first converting to a quaternion
    representation, and then by converting the quaternion to an axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three small Euler angles. `[A1, ..., An, 0]` is the angle
      about `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians
      and `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to
      "axis_angle_from_euler_with_small_angles_approximation".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.
  """
  with tf.name_scope(name):
    quaternion = quaternion_lib.from_euler_with_small_angles_approximation(
        angles)
    return from_quaternion(quaternion)


def from_quaternion(quaternion, name="axis_angle_from_quaternion"):
  """Converts a quaternion to an axis-angle representation.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "axis_angle_from_quaternion".

  Returns:
    Tuple of two tensors of shape `[A1, ..., An, 3]` and `[A1, ..., An, 1]`,
    where the first tensor represents the axis, and the second represents the
    angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  with tf.name_scope(name):
    quaternion = tf.convert_to_tensor(value=quaternion)

    shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    quaternion = asserts.assert_normalized(quaternion)

    # This prevents zero norm xyz and zero w, and is differentiable.
    quaternion += asserts.select_eps_for_addition(quaternion.dtype)
    xyz, w = tf.split(quaternion, (3, 1), axis=-1)
    norm = tf.norm(tensor=xyz, axis=-1, keepdims=True)
    angle = 2.0 * tf.atan2(norm, tf.abs(w))
    axis = safe_ops.safe_unsigned_div(safe_ops.nonzero_sign(w) * xyz, norm)
    return axis, angle


def from_rotation_matrix(rotation_matrix,
                         name="axis_angle_from_rotation_matrix"):
  """Converts a rotation matrix to an axis-angle representation.

  Note:
    In the current version the returned axis-angle representation is not unique
    for a given rotation matrix. Since a direct conversion would not really be
    faster, we first transform the rotation matrix to a quaternion, and finally
    perform the conversion from that quaternion to the corresponding axis-angle
    representation.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "axis_angle_from_rotation_matrix".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
  """
  with tf.name_scope(name):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

    shape.check_static(
        tensor=rotation_matrix,
        tensor_name="rotation_matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))
    rotation_matrix = rotation_matrix_3d.assert_rotation_matrix_normalized(
        rotation_matrix)

    quaternion = quaternion_lib.from_rotation_matrix(rotation_matrix)
    return from_quaternion(quaternion)


def inverse(axis, angle, name="axis_angle_inverse"):
  """Computes the axis-angle that is the inverse of the input axis-angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]` where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_inverse".

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor represents the axis, and the
    second represents the angle. The resulting axis is a normalized vector.

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
        tensors=(axis, angle),
        tensor_names=("axis", "angle"),
        last_axes=-2,
        broadcast_compatible=True)

    axis = asserts.assert_normalized(axis)
    return axis, -angle


def is_normalized(axis, angle, atol=1e-3, name="axis_angle_is_normalized"):
  """Determines if the axis-angle is normalized or not.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]` where the last dimension
      represents an angle.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "axis_angle_is_normalized".

  Returns:
    A tensor of shape `[A1, ..., An, 1]`, where False indicates that the axis is
    not normalized.
  """
  with tf.name_scope(name):
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

    norms = tf.norm(tensor=axis, axis=-1, keepdims=True)
    return tf.abs(norms - 1.) < atol


def rotate(point, axis, angle, name="axis_angle_rotate"):
  r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.

  Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
  $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:

  $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
  +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to rotate.
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_rotate".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If `point`, `axis`, or `angle` are of different shape or if
    their respective shape is not supported.
  """
  with tf.name_scope(name):
    point = tf.convert_to_tensor(value=point)
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 3))
    shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(point, axis, angle),
        tensor_names=("point", "axis", "angle"),
        last_axes=-2,
        broadcast_compatible=True)
    axis = asserts.assert_normalized(axis)

    cos_angle = tf.cos(angle)
    axis_dot_point = vector.dot(axis, point)
    return point * cos_angle + vector.cross(
        axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
