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

import tensorflow.compat.v2 as tf
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
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

    quaternion_real = asserts.assert_normalized(quaternion_real)

    return tf.concat((quaternion.conjugate(quaternion_real),
                      quaternion.conjugate(quaternion_dual)),
                     axis=-1)


def multiply(dual_quaternion1: type_alias.TensorLike,
             dual_quaternion2: type_alias.TensorLike,
             name: str = "dual_quaternion_multiply"):
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


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
