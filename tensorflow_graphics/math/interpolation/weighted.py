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
"""This module implements weighted interpolation for point sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def interpolate(points,
                weights,
                indices,
                normalize=True,
                allow_negative_weights=False,
                name=None):
  """Weighted interpolation for M-D point sets.

  Given an M-D point set, this function can be used to generate a new point set
  that is formed by interpolating a subset of points in the set.

  Note:
    In the following, A1 to An, and B1 to Bk are optional batch dimensions.

  Args:
    points: A tensor with shape `[B1, ..., Bk, M] and rank R > 1, where M is the
      dimensionality of the points.
    weights: A tensor with shape `[A1, ..., An, P]`, where P is the number of
      points to interpolate for each output point.
    indices: A tensor of dtype tf.int32 and shape `[A1, ..., An, P, R-1]`, which
      contains the point indices to be used for each output point. The R-1
      dimensional axis gives the slice index of a single point in `points`. The
      first n+1 dimensions of weights and indices must match, or be broadcast
      compatible.
    normalize: A `bool` describing whether or not to normalize the weights on
      the last axis.
    allow_negative_weights: A `bool` describing whether or not negative weights
      are allowed.
    name: A name for this op. Defaults to "weighted_interpolate".

  Returns:
    A tensor of shape `[A1, ..., An, M]` storing the interpolated M-D
    points. The first n dimensions will be the same as weights and indices.
  """
  with tf.compat.v1.name_scope(name, "weighted_interpolate",
                               [points, weights, indices]):
    points = tf.convert_to_tensor(value=points)
    weights = tf.convert_to_tensor(value=weights)
    indices = tf.convert_to_tensor(value=indices)

    shape.check_static(
        tensor=points, tensor_name="points", has_rank_greater_than=1)
    shape.check_static(
        tensor=indices,
        tensor_name="indices",
        has_rank_greater_than=1,
        has_dim_equals=(-1, points.shape.ndims - 1))
    shape.compare_dimensions(
        tensors=(weights, indices),
        axes=(-1, -2),
        tensor_names=("weights", "indices"))
    shape.compare_batch_dimensions(
        tensors=(weights, indices),
        last_axes=(-2, -3),
        tensor_names=("weights", "indices"),
        broadcast_compatible=True)
    if not allow_negative_weights:
      weights = asserts.assert_all_above(weights, 0.0, open_bound=False)

    if normalize:
      sums = tf.reduce_sum(input_tensor=weights, axis=-1, keepdims=True)
      sums = asserts.assert_nonzero_norm(sums)
      weights = safe_ops.safe_signed_div(weights, sums)
    point_lists = tf.gather_nd(points, indices)
    return vector.dot(
        point_lists, tf.expand_dims(weights, axis=-1), axis=-2, keepdims=False)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
