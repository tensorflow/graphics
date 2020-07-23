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
"""Tangential lens distortion function.

Given a vector in homogeneous coordinates, `(x/z, y/z, 1)`, we define
`r^2 = (x/z)^2 + (y/z)^2`. Let `u = x/z` and `v = y/z`. We use the tangential
distortion functions `f(u) = 2 * p1 * u * v + p2 * (r^2 + 2 * u^2)` and
`f(v) = p1 * (r^2 + 2 * v^2) + 2 * p2 * u * v`. The distorted vector is given by
`(x/z + f(x/z), y/z + f(y/z), 1)`.

TODO: Undistortion
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def distortion_terms(projective_x,
                     projective_y,
                     distortion_coefficient_1,
                     distortion_coefficient_2,
                     name=None):
  """Calculates a tangential distortion terms given normalized image coordinates.

  Given a vector describing a location in camera space in homogeneous
  coordinates, `(x/z, y/z, 1)`, squared_radius is `r^2 = (x/z)^2 + (y/z)^2`.
  distortion_terms are added to `x/z` and `y/z` to obtain the distorted
  coordinates. In this function, `x_distortion_term` is given by
  `2 * distortion_coefficient_1 * projective_x * projective_y +
  distortion coefficient_2 * (squared_radius + 2 * squared_projective_x)`, and
  `y_distortion_term` is given by
  `distortion_coefficient_1 * (squared_radius + 2 * squared_projective_y) +
  2 * distortion_coefficient_2 * projective_x * projective_y`.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    projective_x: A tensor of shape `[A1, ..., An, H, W]`, containing the
      undistorted projective coordinates `(x/z)` of the image pixels.
    projective_y: A tensor of shape `[A1, ..., An, H, W]`, containing the
      undistorted projective coordinates `(y/z)` of the image pixels.
    distortion_coefficient_1: A `scalar` or a tensor of shape `[A1, ..., An]`,
      which contains the first tangential distortion coefficients of each image.
    distortion_coefficient_2: A `scalar` or a tensor of shape `[A1, ..., An]`,
      which contains the second tangential distortion coefficients of each image.
    name: A name for this op. Defaults to
      "tangential_distortion_distortion_terms".

  Returns:
    x_distortion_term: A tensor of shape `[A1, ..., An, H, W]`, the correction
      terms that should be added to the projective coordinates (x/z) to apply
      the distortion along the x-axis of the image.
    y_distortion_term: A tensor of shape `[A1, ..., An, H, W]`, the correction
      terms that should be added to the projective coordinates (y/z) to apply
      the distortion along the y-axis of the image.
    x_overflow_mask: A boolean tensor of shape `[A1, ..., An, H, W]`, `True`
      where `projective_x` is beyond the range where the distortion function is
      monotonically increasing. Wherever `overflow_mask` is True,
      `x_distortion_term`'s value is meaningless.
    y_overflow_mask: A boolean tensor of shape `[A1, ..., An, H, W]`, `True`
      where `projective_y` is beyond the range where the distortion function is
      monotonically increasing. Wherever `overflow_mask` is True,
      `y_distortion_term`'s value is meaningless.
  """
  with tf.compat.v1.name_scope(name,
                               "tangential_distortion_distortion_terms",
                               [projective_x,
                                projective_y,
                                distortion_coefficient_1,
                                distortion_coefficient_2]):
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
      tensor=projective_x,
      tensor_name="projective_x",
      has_rank_greater_than=1)
    shape.check_static(
      tensor=projective_y,
      tensor_name="projective_y",
      has_rank_greater_than=1)
    shape.compare_batch_dimensions(
      tensors=(projective_x,
               projective_y,
               distortion_coefficient_1,
               distortion_coefficient_2),
      last_axes=(-3, -3, -1, -1),
      broadcast_compatible=True)
    projective_x = asserts.assert_all_above(
      projective_x, 0.0, open_bound=False)
    projective_y = asserts.assert_all_above(
      projective_y, 0.0, open_bound=False)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    squared_radius = projective_x ** 2.0 + projective_y ** 2.0
    double_squared_projective_x = 2.0 * projective_x ** 2.0
    double_squared_projective_y = 2.0 * projective_y ** 2.0
    double_distortion_coefficient_1 = 2.0 * distortion_coefficient_1
    double_distortion_coefficient_2 = 2.0 * distortion_coefficient_2
    squared_radius_plus_double_squared_projective_x = (
      squared_radius + double_squared_projective_x)
    squared_radius_plus_double_squared_projective_y = (
      squared_radius + double_squared_projective_y)
    x_distortion_term = (
      double_distortion_coefficient_1 * projective_x * projective_y +
      distortion_coefficient_2 *
      squared_radius_plus_double_squared_projective_x)
    y_distortion_term = (
      distortion_coefficient_1 *
      squared_radius_plus_double_squared_projective_y +
      double_distortion_coefficient_2 * projective_x * projective_y)
    x_overflow_mask = tf.less(
      1.0 + 2.0 * distortion_coefficient_1 * projective_y + 6.0 *
      distortion_coefficient_2 * projective_x, 0.0)
    y_overflow_mask = tf.less(
      1.0 + 2.0 * distortion_coefficient_2 * projective_x + 6.0 *
      distortion_coefficient_1 * projective_y, 0.0)
    return (x_distortion_term,
            y_distortion_term,
            x_overflow_mask,
            y_overflow_mask)


def undistortion_terms(distorted_projective_x,
                       distorted_projective_y,
                       distortion_coefficient_1,
                       distortion_coefficient_2,
                       num_iterations=5,
                       name=None):
  """Calculates the inverse tangential distortion function given distorted
  projective coordinates coordinates.

  Given a vector describing a location in camera space in homogeneous
  coordinates `(x/z, y/z, 1)`, after distortion has been applied, these become
  `(x'/z, y'/z, 1)`. `distorted_projective_x` is `(x'/z)` and
  `distorted_projective_y` is `(y'/z)`. `undistortion_terms` are added to `x'/z`
  and `y'/z` to obtain the undistorted projective coordinates `x/z` and `y/z`.
  The undistortion_factor in this function is derived from a tangential
  distortion function, where the distortion factor for x equals
  `2 * distortion_coefficient_1 * projective_x * projective_y
  + distortion_coefficient_2 * (squared_radius + 2 * projective_x ** 2)`,
  and the distortion factor for y equals
  `2 * distortion_coefficient_2 * projective_x * projective_y
  + distortion_coefficient_1 * (squared_radius + 2 * projective_y ** 2)`.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    distorted_projective_x: A tensor of shape `[A1, ..., An, H, W]` containing
      the value of projective coordinates `x'/z`.
    distorted_projective_y: A tensor of shape `[A1, ..., An, H, W]` containing
      the value of projective coordinates `x'/z`.
    distortion_coefficient_1: A `scalar` or a tensor of shape `[A1, ..., An]`,
      which contains the first tangential distortion coefficients of each image.
    distortion_coefficient_2: A `scalar` or a tensor of shape `[A1, ..., An]`,
      which contains the second tangential distortion coefficients of each image.
    num_iterations: Number of Newton-Raphson iterations to calculate the inverse
      distortion function. Defaults to 5, which is on the high-accuracy side.
    name: A name for this op. Defaults to
      "tangential_distortion_undistortion_terms".

  Returns:
    x_undistortion_term: A tensor of shape `[A1, ..., An, H, W]`, the correction
      terms that should be added to the distorted projective x coordinates,
      `x'/z` to obtain the undistorted ones.
    y_undistortion_term: A tensor of shape `[A1, ..., An, H, W]`, the correction
      terms that should be added to the distorted projective y coordinates,
      `y'/z` to obtain the undistorted ones.
    x_overflow_mask: A boolean tensor of shape `[A1, ..., An, H, W]`, `True`
      where `distorted_projective_x` is beyond the range where the distortion function is
      monotonically increasing. Wherever `overflow_mask` is True,
      `x_undistortion_term`'s value is meaningless.
    y_overflow_mask: A boolean tensor of shape `[A1, ..., An, H, W]`, `True`
      where `distorted_projective_y` is beyond the range where the distortion function is
      monotonically increasing. Wherever `overflow_mask` is True,
      `y_undistortion_terms`'s value is meaningless.
  """
  with tf.compat.v1.name_scope(name,
                               "tangential_distortion_undistortion_terms",
                               [distorted_projective_x,
                                distorted_projective_y,
                                distortion_coefficient_1,
                                distortion_coefficient_2]):
    distorted_projective_x = tf.convert_to_tensor(value=distorted_projective_x)
    distorted_projective_y = tf.convert_to_tensor(value=distorted_projective_y)
    distortion_coefficient_1 = tf.convert_to_tensor(
      value=distortion_coefficient_1)
    distortion_coefficient_2 = tf.convert_to_tensor(
      value=distortion_coefficient_2)

    if distortion_coefficient_1.shape.ndims == 0:
      distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=0)
    if distortion_coefficient_2.shape.ndims == 0:
      distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=0)
    shape.check_static(
      tensor=distorted_projective_x,
      tensor_name="distorted_projective_x",
      has_rank_greater_than=1)
    shape.check_static(
      tensor=distorted_projective_y,
      tensor_name="distorted_projective_y",
      has_rank_greater_than=1)
    shape.compare_batch_dimensions(
      tensors=(distorted_projective_x,
               distorted_projective_y,
               distortion_coefficient_1,
               distortion_coefficient_2),
      last_axes=(-3, -3, -1, -1),
      broadcast_compatible=True)
    distorted_projective_x = asserts.assert_all_above(
      distorted_projective_x, 0.0, open_bound=False)
    distorted_projective_y = asserts.assert_all_above(
      distorted_projective_y, 0.0, open_bound=False)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_1 = tf.expand_dims(distortion_coefficient_1, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    distortion_coefficient_2 = tf.expand_dims(distortion_coefficient_2, axis=-1)
    distorted_squared_radius = (
      distorted_projective_x ** 2.0 + distorted_projective_y ** 2.0)

    return None


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
