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
r"""This modules implements Euler angles functionalities.

The Euler angles are defined using a vector $$[\theta, \gamma, \beta]^T \in
\mathbb{R}^3$$, where $$\theta$$ is the angle about $$x$$, $$\gamma$$ the angle
about $$y$$, and $$\beta$$ is the angle about $$z$$

More details about Euler angles can be found on [this page.]
(https://en.wikipedia.org/wiki/Euler_angles)

Note: The angles are defined in radians.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def from_axis_angle(axis, angle, name="euler_from_axis_angle"):
  """Converts axis-angle to Euler angles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "euler_from_axis_angle".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    the three Euler angles.
  """
  with tf.name_scope(name):
    return from_quaternion(quaternion.from_axis_angle(axis, angle))


def from_quaternion(quaternions, name="euler_from_quaternion"):
  """Converts quaternions to Euler angles.

  Args:
    quaternions: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "euler_from_quaternion".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    the three Euler angles.
  """

  def general_case(r00, r10, r21, r22, r20, eps_addition):
    """Handles the general case."""
    theta_y = -tf.asin(r20)
    sign_cos_theta_y = safe_ops.nonzero_sign(tf.cos(theta_y))
    r00 = safe_ops.nonzero_sign(r00) * eps_addition + r00
    r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
    theta_z = tf.atan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
    theta_x = tf.atan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)
    return tf.stack((theta_x, theta_y, theta_z), axis=-1)

  def gimbal_lock(r01, r02, r20, eps_addition):
    """Handles Gimbal locks."""
    sign_r20 = safe_ops.nonzero_sign(r20)
    r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
    theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
    theta_y = -sign_r20 * tf.constant(math.pi / 2.0, dtype=r20.dtype)
    theta_z = tf.zeros_like(theta_x)
    angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
    return angles

  with tf.name_scope(name):
    quaternions = tf.convert_to_tensor(value=quaternions)

    shape.check_static(
        tensor=quaternions, tensor_name="quaternions", has_dim_equals=(-1, 4))

    x, y, z, w = tf.unstack(quaternions, axis=-1)
    tx = safe_ops.safe_shrink(2.0 * x, -2.0, 2.0, True)
    ty = safe_ops.safe_shrink(2.0 * y, -2.0, 2.0, True)
    tz = safe_ops.safe_shrink(2.0 * z, -2.0, 2.0, True)
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    # The following is clipped due to numerical instabilities that can take some
    # enties outside the [-1;1] range.
    r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
    r10 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
    r21 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)
    r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
    r20 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)
    r01 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
    r02 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)
    eps_addition = asserts.select_eps_for_addition(quaternions.dtype)
    general_solution = general_case(r00, r10, r21, r22, r20, eps_addition)
    gimbal_solution = gimbal_lock(r01, r02, r20, eps_addition)
    # The general solution is unstable close to the Gimbal lock, and the gimbal
    # solution is not toooff in these cases.
    is_gimbal = tf.less(tf.abs(tf.abs(r20) - 1.0), 1.0e-6)
    gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
    return tf.where(gimbal_mask, gimbal_solution, general_solution)


def from_rotation_matrix(rotation_matrix, name="euler_from_rotation_matrix"):
  """Converts rotation matrices to Euler angles.

  The rotation matrices are assumed to have been constructed by rotation around
  the $$x$$, then $$y$$, and finally the $$z$$ axis.

  Note:
    There is an infinite number of solutions to this problem. There are
  Gimbal locks when abs(rotation_matrix(2,0)) == 1, which are not handled.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
      dimensions represent a rotation matrix.
    name: A name for this op that defaults to "euler_from_rotation_matrix".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    the three Euler angles.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
  """

  def general_case(rotation_matrix, r20, eps_addition):
    """Handles the general case."""
    theta_y = -tf.asin(r20)
    sign_cos_theta_y = safe_ops.nonzero_sign(tf.cos(theta_y))
    r00 = rotation_matrix[..., 0, 0]
    r10 = rotation_matrix[..., 1, 0]
    r21 = rotation_matrix[..., 2, 1]
    r22 = rotation_matrix[..., 2, 2]
    r00 = safe_ops.nonzero_sign(r00) * eps_addition + r00
    r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
    # cos_theta_y evaluates to 0 on Gimbal locks, in which case the output of
    # this function will not be used.
    theta_z = tf.atan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
    theta_x = tf.atan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)
    angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
    return angles

  def gimbal_lock(rotation_matrix, r20, eps_addition):
    """Handles Gimbal locks."""
    r01 = rotation_matrix[..., 0, 1]
    r02 = rotation_matrix[..., 0, 2]
    sign_r20 = safe_ops.nonzero_sign(r20)
    r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
    theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
    theta_y = -sign_r20 * tf.constant(math.pi / 2.0, dtype=r20.dtype)
    theta_z = tf.zeros_like(theta_x)
    angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
    return angles

  with tf.name_scope(name):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

    shape.check_static(
        tensor=rotation_matrix,
        tensor_name="rotation_matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3), (-2, 3)))
    rotation_matrix = rotation_matrix_3d.assert_rotation_matrix_normalized(
        rotation_matrix)

    r20 = rotation_matrix[..., 2, 0]
    eps_addition = asserts.select_eps_for_addition(rotation_matrix.dtype)
    general_solution = general_case(rotation_matrix, r20, eps_addition)
    gimbal_solution = gimbal_lock(rotation_matrix, r20, eps_addition)
    is_gimbal = tf.equal(tf.abs(r20), 1)
    gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
    return tf.where(gimbal_mask, gimbal_solution, general_solution)


def inverse(euler_angle, name="euler_inverse"):
  """Computes the angles that would inverse a transformation by euler_angle.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    euler_angle: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles.
    name: A name for this op that defaults to "euler_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    the three Euler angles.

  Raises:
    ValueError: If the shape of `euler_angle` is not supported.
  """
  with tf.name_scope(name):
    euler_angle = tf.convert_to_tensor(value=euler_angle)

    shape.check_static(
        tensor=euler_angle, tensor_name="euler_angle", has_dim_equals=(-1, 3))

    return -euler_angle


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
