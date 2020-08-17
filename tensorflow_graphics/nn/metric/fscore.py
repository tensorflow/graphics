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
"""This module implements the fscore metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.nn.metric import precision as precision_module
from tensorflow_graphics.nn.metric import recall as recall_module
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def evaluate(ground_truth,
             prediction,
             precision_function=precision_module.evaluate,
             recall_function=recall_module.evaluate,
             name=None):
  """Computes the fscore metric for the given ground truth and predicted labels.

  The fscore is calculated as 2 * (precision * recall) / (precision + recall)
  where the precision and recall are evaluated by the given function parameters.
  The precision and recall functions default to their definition for boolean
  labels (see https://en.wikipedia.org/wiki/Precision_and_recall for more
  details).

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    ground_truth: A tensor of shape `[A1, ..., An, N]`, where the last axis
      represents the ground truth values.
    prediction: A tensor of shape `[A1, ..., An, N]`, where the last axis
      represents the predicted values.
    precision_function: The function to use for evaluating the precision.
      Defaults to the precision evaluation for binary ground-truth and
      predictions.
    recall_function: The function to use for evaluating the recall. Defaults to
      the recall evaluation for binary ground-truth and prediction.
    name: A name for this op. Defaults to "fscore_evaluate".

  Returns:
    A tensor of shape `[A1, ..., An]` that stores the fscore metric for the
    given ground truth labels and predictions.

  Raises:
    ValueError: if the shape of `ground_truth`, `prediction` is
    not supported.
  """
  with tf.compat.v1.name_scope(name, "fscore_evaluate",
                               [ground_truth, prediction]):
    ground_truth = tf.convert_to_tensor(value=ground_truth)
    prediction = tf.convert_to_tensor(value=prediction)

    shape.compare_batch_dimensions(
        tensors=(ground_truth, prediction),
        tensor_names=("ground_truth", "prediction"),
        last_axes=-1,
        broadcast_compatible=True)

    recall = recall_function(ground_truth, prediction)
    precision = precision_function(ground_truth, prediction)

    return safe_ops.safe_signed_div(2 * precision * recall, precision + recall)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
