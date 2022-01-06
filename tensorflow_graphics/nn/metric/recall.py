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
"""This module implements the recall metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, List, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def _cast_to_int(prediction):
  return tf.cast(x=prediction, dtype=tf.int32)


def evaluate(ground_truth: type_alias.TensorLike,
             prediction: type_alias.TensorLike,
             classes: Optional[Union[int, List[int], Tuple[int]]] = None,
             reduce_average: bool = True,
             prediction_to_category_function: Callable[..., Any] = _cast_to_int,
             name: str = "recall_evaluate") -> tf.Tensor:
  """Computes the recall metric for the given ground truth and predictions.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    ground_truth: A tensor of shape `[A1, ..., An, N]`, where the last axis
      represents the ground truth labels. Will be cast to int32.
    prediction: A tensor of shape `[A1, ..., An, N]`, where the last axis
      represents the predictions (which can be continuous).
    classes: An integer or a list/tuple of integers representing the classes for
      which the recall will be evaluated. In case 'classes' is 'None', the
      number of classes will be inferred from the given values and the recall
      will be calculated for each of the classes. Defaults to 'None'.
    reduce_average: Whether to calculate the average of the recall for each
      class and return a single recall value. Defaults to true.
    prediction_to_category_function: A function to associate a `prediction` to a
      category. Defaults to rounding down the value of the prediction to the
      nearest integer value.
    name: A name for this op. Defaults to "recall_evaluate".

  Returns:
    A tensor of shape `[A1, ..., An, C]`, where the last axis represents the
    recall calculated for each of the requested classes.

  Raises:
    ValueError: if the shape of `ground_truth`, `prediction` is not supported.
  """
  with tf.name_scope(name):
    ground_truth = tf.cast(
        x=tf.convert_to_tensor(value=ground_truth), dtype=tf.int32)
    prediction = tf.convert_to_tensor(value=prediction)

    shape.compare_batch_dimensions(
        tensors=(ground_truth, prediction),
        tensor_names=("ground_truth", "prediction"),
        last_axes=-1,
        broadcast_compatible=True)

    prediction = prediction_to_category_function(prediction)
    if classes is None:
      num_classes = tf.math.maximum(
          tf.math.reduce_max(input_tensor=ground_truth),
          tf.math.reduce_max(input_tensor=prediction)) + 1
      classes = tf.range(num_classes)
    else:
      classes = tf.convert_to_tensor(value=classes)
      # Make sure classes is a tensor of rank 1.
      classes = tf.reshape(classes, [1]) if tf.rank(classes) == 0 else classes

    # Create a confusion matrix for each of the classes (with dimensions
    # [A1, ..., An, C, N]).
    classes = tf.expand_dims(classes, -1)
    ground_truth_per_class = tf.equal(tf.expand_dims(ground_truth, -2), classes)
    prediction_per_class = tf.equal(tf.expand_dims(prediction, -2), classes)

    # Caluclate the recall for each of the classes.
    true_positives = tf.math.reduce_sum(
        input_tensor=tf.cast(
            x=tf.math.logical_and(ground_truth_per_class, prediction_per_class),
            dtype=tf.float32),
        axis=-1)
    total_ground_truth_positives = tf.math.reduce_sum(
        input_tensor=tf.cast(x=ground_truth_per_class, dtype=tf.float32),
        axis=-1)

    recall_per_class = safe_ops.safe_signed_div(true_positives,
                                                total_ground_truth_positives)
    if reduce_average:
      return tf.math.reduce_mean(input_tensor=recall_per_class, axis=-1)
    else:
      return recall_per_class


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
