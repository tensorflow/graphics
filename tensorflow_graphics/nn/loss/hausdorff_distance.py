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
"""This module implements the hausdorff distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def evaluate(point_set_a, point_set_b, name=None):
  """Computes the Hausdorff distance from point_set_a to point_set_b.

  Note:
    Hausdorff distance from point_set_a to point_set_b is defined as the maximum
    of all distances from a point in point_set_a to the closest point in
    point_set_b. It is an asymmetric metric.

  Note:
    This function returns the exact Hausdorff distance and not an approximation.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
      represents points in a D dimensional space.
    point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
      represents points in a D dimensional space.
    name: A name for this op. Defaults to "hausdorff_distance_evaluate".

  Returns:
    A tensor of shape `[A1, ..., An]` storing the hausdorff distance from
    from point_set_a to point_set_b.

  Raises:
    ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
  """
  with tf.compat.v1.name_scope(name, "hausdorff_distance_evaluate",
                               [point_set_a, point_set_b]):
    point_set_a = tf.convert_to_tensor(value=point_set_a)
    point_set_b = tf.convert_to_tensor(value=point_set_b)

    shape.compare_batch_dimensions(
        tensors=(point_set_a, point_set_b),
        tensor_names=("point_set_a", "point_set_b"),
        last_axes=-3,
        broadcast_compatible=True)
    # Verify that the last axis of the tensors has the same dimension.
    dimension = point_set_a.shape.as_list()[-1]
    shape.check_static(
        tensor=point_set_b,
        tensor_name="point_set_b",
        has_dim_equals=(-1, dimension))

    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
    # dimension D).
    difference = (
        tf.expand_dims(point_set_a, axis=-2) -
        tf.expand_dims(point_set_b, axis=-3))
    # Calculate the square distances between each two points: |ai - bj|^2.
    square_distances = tf.einsum("...i,...i->...", difference, difference)

    minimum_square_distance_a_to_b = tf.reduce_min(square_distances, axis=-1)
    return tf.sqrt(tf.reduce_max(minimum_square_distance_a_to_b, axis=-1))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
