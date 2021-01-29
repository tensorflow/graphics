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
quaternions $$q_r$$ and $$q_d$$.

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


def conjugate(dual_quaternion, name=None):
  """Computes the conjugate of a dual quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    dual_quaternion: A tensor of shape `[A1, ..., An, 8]`, where the last
    dimension represents a normalized dual quaternion.
    name: A name for this op that defaults to "dual_quaternion_conjugate".

  Returns:
    A tensor of shape `[A1, ..., An, 8]`, where the last dimension represents
    a normalized dual quaternion.

  Raises:
    ValueError: If the shape of `dual_quaternion` is not supported.
  """
  with tf.compat.v1.name_scope(name, "dual_quaternion_conjugate",
                               [dual_quaternion]):
    dual_quaternion = tf.convert_to_tensor(value=dual_quaternion)

    shape.check_static(
        tensor=dual_quaternion, tensor_name="dual_quaternion",
        has_dim_equals=(-1, 8))

    quaternion_real, quaternion_dual = tf.split(
        dual_quaternion, (4, 4), axis=-1)

    quaternion_real = asserts.assert_normalized(quaternion_real)

    return tf.concat((quaternion.conjugate(quaternion_real),
                      quaternion.conjugate(quaternion_dual)),
                     axis=-1)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
