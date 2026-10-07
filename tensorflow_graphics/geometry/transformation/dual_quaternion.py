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
"""This module implements TensorFlow dual quaternion utility functions.

A dual quaternion is an extension of a quaternion with the real and dual parts
and written as $$q = q_r + epsilon q_d$$, where $$epsilon$$ is the dual number
with the property $$e^2 = 0$$. It can thus be represented as two quaternions,
and thus stored as 8 numbers. We define the operations in terms of the two
quaternions $$q_r$$ and $$q_d$$, which are stored as 8-dimensional tensor.

Dual quaternions are extensions of quaternions to represent rigid
transformations (rotations and translations). They are in particular important
for deforming geometries as linear blending is a very close approximation of
closest path blending, which is not the case for any other representation.

Note: Some of the functions expect normalized quaternions as inputs where
$$|q_r| = 1$$.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def conjugate(dual_quaternion: type_alias.TensorLike,
              name: str = "dual_quaternion_conjugate") -> tf.Tensor:
  """Computes the conjugate of a dual quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion: A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a normalized dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_conjugate".

  Returns:
    A tensor of shape `[A1, ..., An, 8]`, where the last dimension represents
    a normalized dual quaternion.

  Raises:
    ValueError: If the shape of `dual_quaternion` is not supported.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    quaternion_real, quaternion_dual = tf.split(
        dual_quaternion, (4, 4), axis=-1)

    return tf.concat((quaternion.conjugate(quaternion_real),
                      quaternion.conjugate(quaternion_dual)),
                     axis=-1)


def multiply(dual_quaternion1: type_alias.TensorLike,
             dual_quaternion2: type_alias.TensorLike,
             name: str = "dual_quaternion_multiply") -> tf.Tensor:
  """Multiplies two dual quaternions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion1:  A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a dual quaternion.
    dual_quaternion2:  A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_multiply".

  Returns:
    A tensor of shape `[A1, ..., An, 8]` representing dual quaternions.
  """
  with tf.name_scope(name):
    dual_quaternion1 = tf.convert_to_tensor(value=dual_quaternion1)
    dual_quaternion2 = tf.convert_to_tensor(value=dual_quaternion2)

    shape.check_static(
        tensor=dual_quaternion1,
        tensor_name="dual_quaternion1",
        has_dim_equals=(-1, 8))
    shape.check_static(
        tensor=dual_quaternion2,
        tensor_name="dual_quaternion2",
        has_dim_equals=(-1, 8))

    dual_quaternion1_real, dual_quaternion1_dual = tf.split(
        dual_quaternion1, (4, 4), axis=-1)
    dual_quaternion2_real, dual_quaternion2_dual = tf.split(
        dual_quaternion2, (4, 4), axis=-1)

    dual_quaternion_output_real = quaternion.multiply(dual_quaternion1_real,
                                                      dual_quaternion2_real)
    dual_quaternion_output_dual = (
        quaternion.multiply(dual_quaternion1_real, dual_quaternion2_dual) +
        quaternion.multiply(dual_quaternion1_dual, dual_quaternion2_real))

    return tf.concat((dual_quaternion_output_real, dual_quaternion_output_dual),
                     axis=-1)


def inverse(dual_quaternion: type_alias.TensorLike,
            name: str = "dual_quaternion_inverse") -> tf.Tensor:
  """Computes the inverse of a dual quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion:  A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 8]`, where the last dimension represents
    a dual quaternion.

  Raises:
    ValueError: If the shape of `dual quaternion` is not supported.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    quaternion_real, quaternion_dual = tf.split(
        dual_quaternion, (4, 4), axis=-1)

    quaternion_real_norm_squared = tf.norm(
        tensor=quaternion_real, axis=-1, keepdims=True) ** 2
    quaternion_real_conj = quaternion.conjugate(quaternion_real)

    quaternion_output_real = safe_ops.safe_signed_div(
        quaternion_real_conj,
        quaternion_real_norm_squared)

    normalized_dual = safe_ops.safe_signed_div(
        quaternion.conjugate(quaternion_dual), quaternion_real_norm_squared)
    normalized_dot_product = safe_ops.safe_signed_div(
        vector.dot(quaternion_real, quaternion_dual, keepdims=True),
        quaternion_real_norm_squared**2)
    quaternion_output_dual = (
        normalized_dual - 2 * quaternion_real_conj * normalized_dot_product)

    return tf.concat((quaternion_output_real, quaternion_output_dual), axis=-1)


def norm(dual_quaternion: type_alias.TensorLike,
         name: str = "dual_quaternion_inverse") -> tf.Tensor:
  """Computes the norm, which is in general a dual number.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion:  A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_inverse".

  Returns:
    A tensor of shape `[A1, ..., An, 2]`, where the last dimension represents
    a dual number.

  Raises:
    ValueError: If the shape of `dual quaternion` is not supported.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    quaternion_real, quaternion_dual = tf.split(
        dual_quaternion, (4, 4), axis=-1)

    quaternion_real_norm = tf.norm(
        tensor=quaternion_real, axis=-1, keepdims=True)
    normalized_dot_product = safe_ops.safe_signed_div(
        vector.dot(quaternion_real, quaternion_dual, keepdims=True),
        quaternion_real_norm)

    return tf.concat((quaternion_real_norm, normalized_dot_product), axis=-1)


def is_normalized(dual_quaternion: type_alias.TensorLike,
                  atol: tf.float32 = 1e-3,
                  name: str = "dual_quaternion_is_normalized") -> bool:
  """Determines if a dual quaternion is normalized or not.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion:  A `[A1, ..., An, 8]`-tensor, where the last dimension
      represents a dual quaternion.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "dual_quaternion_is_normalized".

  Returns:
    A `[A1, ..., An, 1]`-tensor of type `bool`, where False indicates that the
    dual quaternion is not normalized.

  Raises:
    ValueError: If the shape of `dual_quaternion` is not supported.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    norms = norm(dual_quaternion)

    return tf.expand_dims(
        tf.math.logical_and(
            tf.abs(norms[..., 0] - 1.) < atol,
            tf.abs(norms[..., 1] - 0.) < atol),
        axis=-1)


def from_rotation_translation(
    rotation_quaternion: type_alias.TensorLike,
    translation_vector: type_alias.TensorLike,
    name: str = "dual_quaternion_from_rotation_translation") -> tf.Tensor:
  """Converts a rotation matrix and translation vector to a dual quaternion.

  Warning:
    This function is not smooth everywhere.

  Note:
    In the following, A1 to An are optional batch dimensions. Rotation is
    applied first.

  Args:
    rotation_quaternion: A `[A1, ..., An, 4]`-tensor, where the last dimension
      represents a rotation in the form a quaternion.
    translation_vector: A `[A1, ..., An, 3]`-tensor, where the last dimension
      represents a translation vector.
    name: A name for this op that defaults to "dual_quaternion_from_rot_trans".

  Returns:
    A `[A1, ..., An, 8]`-tensor, where the last dimension represents a
    normalized dual quaternion.

  Raises:
    ValueError: If the shape of `rotation_matrix` is not supported.
  """
  with tf.name_scope(name):
    rotation_quaternion = tf.convert_to_tensor(value=rotation_quaternion)
    translation_vector = tf.convert_to_tensor(value=translation_vector)

    shape.check_static(
        tensor=rotation_quaternion,
        tensor_name="rotation_quaternion",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 4))

    shape.check_static(
        tensor=translation_vector,
        tensor_name="translation_vector",
        has_dim_equals=(-1, 3))

    scalar_shape = tf.concat((tf.shape(translation_vector)[:-1], (1,)), axis=-1)
    dtype = translation_vector.dtype

    quaternion_translation = tf.concat(
        (translation_vector, tf.zeros(scalar_shape, dtype)), axis=-1)

    dual_quaternion_dual_part = 0.5 * quaternion.multiply(
        quaternion_translation, rotation_quaternion)

    return tf.concat((rotation_quaternion, dual_quaternion_dual_part), axis=-1)


def to_rotation_translation(
    dual_quaternion: type_alias.TensorLike,
    name: str = "dual_quaternion_to_rot_trans") -> Tuple[tf.Tensor, tf.Tensor]:
  """Converts a dual quaternion into a quaternion for rotation and translation.

  Args:
    dual_quaternion: A `[A1, ..., An, 8]`-tensor, where the last dimension
      represents a qual quaternion.
    name: A name for this op that defaults to "dual_quaternion_to_rot_trans".

  Returns:
    A tuple with a `[A1, ..., An, 4]`-tensor for rotation in quaternion form,
    and a `[A1, ..., An, 3]`-tensor for translation, in that order.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    rotation = dual_quaternion[..., 0:4]
    translation = 2 * quaternion.multiply(
        dual_quaternion[..., 4:8], quaternion.inverse(rotation))
    translation = translation[..., 0:3]

    return rotation, translation


def from_axis_angle_translation(axis: type_alias.TensorLike,
                                angle: type_alias.TensorLike,
                                translation_vector: type_alias.TensorLike,
                                name: str = "dual_quat_from_axis_angle_trans"
                                ) -> type_alias.TensorLike:
  """Converts an axis-angle rotation and translation to a dual quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    translation_vector: A `[A1, ..., An, 3]`-tensor, where the last dimension
      represents a translation vector.
    name: A name for this op that defaults to "dual_quat_from_axis_angle_trans".

  Returns:
    A `[A1, ..., An, 8]`-tensor, where the last dimension represents a
    normalized dual quaternion.

  Raises:
    ValueError: If the shape of `axis`, `angle`, or `translation_vector`
    is not supported.
  """
  with tf.name_scope(name):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)
    translation_vector = tf.convert_to_tensor(value=translation_vector)

    shape.check_static(tensor=axis,
                       tensor_name="axis",
                       has_dim_equals=(-1, 3))
    shape.check_static(tensor=angle,
                       tensor_name="angle",
                       has_dim_equals=(-1, 1))
    shape.check_static(tensor=translation_vector,
                       tensor_name="translation_vector",
                       has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(tensors=(axis, angle, translation_vector),
                                   last_axes=-2,
                                   broadcast_compatible=True)

    scalar_shape = tf.concat((tf.shape(translation_vector)[:-1], (1,)), axis=-1)
    dtype = translation_vector.dtype

    quaternion_rotation = quaternion.from_axis_angle(axis, angle)
    quaternion_translation = tf.concat(
        (translation_vector, tf.zeros(scalar_shape, dtype)), axis=-1)

    dual_quaternion_dual_part = 0.5 * quaternion.multiply(
        quaternion_translation, quaternion_rotation)

    return tf.concat((quaternion_rotation, dual_quaternion_dual_part), axis=-1)


def conjugate_dual(
    dual_quaternion: type_alias.TensorLike,
    name: str = "dual_quaternion_conjugate") -> tf.Tensor:
  """Computes the conjugate (of dual numbers) in a dual quaternion.

  Note:
    For a dual quaternion q = q_0 + epsilon q_e, the dual conjugate is defined
    as q = q_0 - epsilon q_e.
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion: A TensorLike of shape `[A1, ..., An, 8]`, where the last
      dimension represents a normalized dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_conjugate".

  Returns:
    A tensor of shape `[A1, ..., An, 8]`, where the last dimension represents
    a normalized dual quaternion.

  Raises:
    ValueError: If the shape of `dual_quaternion` is not supported.
  """
  with tf.name_scope(name):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion,
        tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    quaternion_real, quaternion_dual = tf.split(
        dual_quaternion, (4, 4), axis=-1)

    return tf.concat((quaternion_real, -quaternion_dual), axis=-1)


def point_to_dual_quaternion(
    point: type_alias.TensorLike,
    name: str = "dual_quaternion_conjugate") -> tf.Tensor:
  """Converts a 3D point to its dual quaternion representation.

  Args:
    point: A TensorLike of shape `[A1, ..., An, 3]`, where the last
      dimension represents a point.
    name: A name for this op that defaults to "point_to_dual_quaternion".

  Returns:
    The dual quaternion representation of `point`.
  """
  with tf.name_scope(name):
    point = tf.convert_to_tensor(value=point)

    shape.check_static(
        tensor=point,
        tensor_name="point",
        has_dim_equals=(-1, 3))

    ones_vector = tf.ones_like(point)[..., 0:1]

    return tf.concat(
        (ones_vector,
         tf.zeros(tf.concat((tf.shape(point)[0:-1], [4]), -1),
                  dtype=point.dtype),
         point), -1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
