#Copyright 2018 Google LLC
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
"""This module implements math routines used by OpenGL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def perspective_right_handed(vertical_field_of_view,
                             aspect_ratio,
                             near,
                             far,
                             name=None):
  """Generates the matrix for a right handed perspective projection.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertical_field_of_view: A tensor of shape `[A1, ..., An]`, where the last
      dimension represents the vertical field of view of the frustum expressed
      in radians. Note that values for `vertical_field_of_view` must be in the
      range (0,pi).
    aspect_ratio: A tensor of shape `[A1, ..., An, C]`, where the last dimension
      stores the width over height ratio of the frustum. Note that values for
      `aspect_ratio` must be non-negative.
    near:  A tensor of shape `[A1, ..., An, C]`, where the last dimension
      captures the distance between the viewer and the near clipping plane. Note
      that values for `near` must be non-negative.
    far:  A tensor of shape `[A1, ..., An, C]`, where the last dimension
      captures the distance between the viewer and the far clipping plane. Note
      that values for `far` must be greater than those of `near`.
    name: A name for this op. Defaults to 'perspective_rh'.

  Raises:
    InvalidArgumentError: if any input contains data not in the specified range
      of valid values.
    ValueError: if the all the inputs are not of the same shape.

  Returns:
    A tensor of shape `[A1, ..., An, 4, 4]`, containing matrices of right
    handed perspective-view frustum.
  """
  with tf.compat.v1.name_scope(
      name, "perspective_rh",
      [vertical_field_of_view, aspect_ratio, near, far]):
    vertical_field_of_view = tf.convert_to_tensor(value=vertical_field_of_view)
    aspect_ratio = tf.convert_to_tensor(value=aspect_ratio)
    near = tf.convert_to_tensor(value=near)
    far = tf.convert_to_tensor(value=far)

    shape.compare_batch_dimensions(
        tensors=(vertical_field_of_view, aspect_ratio, near, far),
        last_axes=-1,
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

    x = tf.stack((inverse_tan_half_vertical_field_of_view / aspect_ratio, zero,
                  zero, zero),
                 axis=-1)
    y = tf.stack((zero, inverse_tan_half_vertical_field_of_view, zero, zero),
                 axis=-1)
    near_minus_far = near - far
    z = tf.stack(
        (zero, zero,
         (far + near) / near_minus_far, 2.0 * far * near / near_minus_far),
        axis=-1)
    w = tf.stack((zero, zero, -one, zero), axis=-1)

    return tf.stack((x, y, z, w), axis=-2)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
