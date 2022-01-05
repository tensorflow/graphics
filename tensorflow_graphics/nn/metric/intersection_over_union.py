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
"""This module implements the intersection-over-union metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def evaluate(ground_truth_labels: type_alias.TensorLike,
             predicted_labels: type_alias.TensorLike,
             grid_size: int = 1,
             name: str = "intersection_over_union_evaluate") -> tf.Tensor:
  """Computes the Intersection-Over-Union metric for the given ground truth and predicted labels.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible, and G1 to Gm are the grid dimensions.

  Args:
    ground_truth_labels: A tensor of shape `[A1, ..., An, G1, ..., Gm]`, where
      the last m axes represent a grid of ground truth attributes. Each
      attribute can either be 0 or 1.
    predicted_labels: A tensor of shape `[A1, ..., An, G1, ..., Gm]`, where the
      last m axes represent a grid of predicted attributes. Each attribute can
      either be 0 or 1.
    grid_size: The number of grid dimensions. Defaults to 1.
    name: A name for this op. Defaults to "intersection_over_union_evaluate".

  Returns:
    A tensor of shape `[A1, ..., An]` that stores the intersection-over-union
    metric of the given ground truth labels and predictions.

  Raises:
    ValueError: if the shape of `ground_truth_labels`, `predicted_labels` is
    not supported.
  """
  with tf.name_scope(name):
    ground_truth_labels = tf.convert_to_tensor(value=ground_truth_labels)
    predicted_labels = tf.convert_to_tensor(value=predicted_labels)

    shape.compare_batch_dimensions(
        tensors=(ground_truth_labels, predicted_labels),
        tensor_names=("ground_truth_labels", "predicted_labels"),
        last_axes=-grid_size,
        broadcast_compatible=True)

    ground_truth_labels = asserts.assert_binary(ground_truth_labels)
    predicted_labels = asserts.assert_binary(predicted_labels)

    sum_ground_truth = tf.math.reduce_sum(
        input_tensor=ground_truth_labels, axis=list(range(-grid_size, 0)))
    sum_predictions = tf.math.reduce_sum(
        input_tensor=predicted_labels, axis=list(range(-grid_size, 0)))
    intersection = tf.math.reduce_sum(
        input_tensor=ground_truth_labels * predicted_labels,
        axis=list(range(-grid_size, 0)))
    union = sum_ground_truth + sum_predictions - intersection

    return tf.where(
        tf.math.equal(union, 0), tf.ones_like(union), intersection / union)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
