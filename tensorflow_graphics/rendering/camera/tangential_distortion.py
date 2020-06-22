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
"""Tangential lens distortion and un-distortion functions.

TODO
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape

def distortion_terms(squared_radius,
                     projective_x,
                     projective_y,
                     distortion_coefficient_1,
                     distortion_coefficient_2,
                     name=None):
  """Calculates a tangential distortion terms given normalized image coordinates.
  TODO:
  """
  with tf.compat.v1.name_scope(name,
                               "tangential_distortion_distortion_term",
                               [squared_radius,
                                projective_x,
                                projective_y,
                                distortion_coefficient_1,
                                distortion_coefficient_2]):
    squared_radius = tf.convert_to_tensor(value=squared_radius)
    projective_x = tf.convert_to_tensor(value=projective_x)
    projective_y = tf.convert_to_tensor(value=projective_y)
    distortion_coefficient_1 = tf.convert_to_tensor(
      value=distortion_coefficient_1)
    distortion_coefficient_2 = tf.convert_to_tensor(
      value=distortion_coefficient_2)

    if distortion_coefficient_1.shape.ndims == 0:
      distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=0)
    if distortion_coefficient_2.shape.ndims == 0:
      distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=0)
    shape.check_static(
      tensor=squared_radius,
      tensor_name="squared_radius",
      has_rank_greater_than=1)
    shape.check_static(
      tensor=projective_x,
      tensor_name="projective_x",
      has_rank_greater_than=1)
    shape.check_static(
      tensor=projective_y,
      tensor_name="projective_y",
      has_rank_greater_than=1)
    shape.compare_batch_dimensions(
      tensors=(squared_radius,
               projective_x,
               projective_y,
               distortion_coefficient_1,
               distortion_coefficient_2),
      last_axes=(-3, -3, -3, -1, -1),
      broadcast_compatible=True)
    squared_radius = asserts.assert_all_above(
      squared_radius, 0.0, open_bound=False)
    projective_x = asserts.assert_all_above(
      projective_x, 0.0, open_bound=False)
    projective_y = asserts.assert_all_above(
      projective_y, 0.0, open_bound=False)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    double_squared_projective_x = 2.0 * projective_x ** 2.0
    double_squared_projective_y = 2.0 * projective_y ** 2.0
    double_distortion_coefficient_1 = 2.0 * distortion_coefficient_1
    double_distortion_coefficient_2 = 2.0 * distortion_coefficient_2
    squared_radius_plus_double_squared_projective_x = (
      squared_radius + double_squared_projective_x)
    squared_radius_plus_double_squared_projective_y = (
      squared_radius + double_squared_projective_y)
    projective_x_distortion_term_ = (
      double_distortion_coefficient_1 * projective_x * projective_y +
      distortion_coefficient_2 *
      squared_radius_plus_double_squared_projective_x)
    projective_y_distortion_term_ = (
      distortion_coefficient_1 *
      squared_radius_plus_double_squared_projective_y +
      double_distortion_coefficient_2 * projective_x * projective_y)
    # TODO: overflow_mask
    return projective_x_distortion_term_, projective_y_distortion_term_

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
